import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torch.utils.hooks import RemovableHandle
from models.modules import CustomLinear, CustomConv2d, CustomBatchNorm2d
from typing import Tuple, List, NamedTuple, Mapping, Optional, Callable
from tqdm.auto import tqdm, trange
from utils import icycle, MultiEpochsDataLoader
from dataclasses import dataclass
from collections import OrderedDict
from torch import optim


@dataclass
class EKFACState:
    Q_A: torch.Tensor = None
    Q_S: torch.Tensor = None
    scale: torch.Tensor = None # Hessian eigenvalues


class EKFACRegularizer:

    def __init__(self, model: nn.Module, criterion: nn.Module, modules: List[nn.Module]) -> None:
        self.model = model
        self.criterion = criterion
        self.modules : List[nn.Module] = modules
        self.a_dict : Mapping[nn.Module, torch.Tensor] = OrderedDict()
        self.g_dict : Mapping[nn.Module, torch.Tensor] = OrderedDict()
        self.kfac_state_dict : Mapping[nn.Module, KFACState] = OrderedDict()
        self.n_iter = 0
        self._init_kfac_states()
        self.ekfac_state_dict: Mapping[nn.Module, EKFACState] = OrderedDict()

    def _init_kfac_states(self) -> None:
        for module in self.modules:
            if isinstance(module, nn.Linear):
                a_size = module.in_features
                g_size = module.out_features
            elif isinstance(module, nn.Conv2d):
                a_size = module.in_channels * module.kernel_size[0] * module.kernel_size[1]
                g_size = module.out_channels
            elif isinstance(module, nn.BatchNorm2d):
                a_size = g_size = module.num_features
            else:
                raise NotImplementedError()

            if module.bias is not None:
                a_size += 1

            self.kfac_state_dict[module] = KFACState(
                S=module.weight.new_zeros((g_size, g_size)),
                A=module.weight.new_zeros((a_size, a_size))
            )

    def _init_ekfac_states(self) -> None:
        for module in self.modules:
            kfac_state = self.kfac_state_dict[module]
            self.ekfac_state_dict[module] = EKFACState(
                Q_A=torch.symeig(kfac_state.A, eigenvectors=True).eigenvectors,
                Q_S=torch.symeig(kfac_state.S, eigenvectors=True).eigenvectors,
                scale=kfac_state.A.new_zeros((kfac_state.S.size(0), kfac_state.A.size(0))),
            )

    def _del_temp_states(self) -> None:
        del self.a_dict
        del self.g_dict

    def _register_hooks(self) -> List[RemovableHandle]:
        hook_handles = []
        for module in self.modules:
            handle = module.register_forward_hook(self._forward_hook)
            hook_handles.append(handle)
        return hook_handles

    def _remove_hooks(self, hook_handles: List[RemovableHandle]) -> None:
        for handle in hook_handles:
            handle.remove()
        hook_handles.clear()

    @torch.no_grad()
    def _forward_hook(self, module: nn.Module, input: Tuple[torch.Tensor, ...], output: Tuple[torch.Tensor, ...]) -> None:
        if type(module) in (nn.Linear, nn.Conv2d, nn.BatchNorm2d):
            input, = input
            output = output
        elif type(module) in (CustomLinear, CustomConv2d, CustomBatchNorm2d):
            input, _ = input # primal input
            _, output = output # tangent output (=jvp)
        else:
            raise NotImplementedError
        self.a_dict[module] = input.detach()
        def _tensor_backward_hook(grad: torch.Tensor) -> None:
            self.g_dict[module] = grad.detach()
        output.requires_grad_(True).register_hook(_tensor_backward_hook)

    @torch.no_grad()
    def _accumulate_curvature_step(self) -> None:
        for module in self.modules:
            a = self.a_dict[module]
            g = self.g_dict[module]
            kfac_state = self.kfac_state_dict[module]
            kfac_state.A.add_(compute_A(module, a))
            kfac_state.S.add_(compute_S(module, g))
        self.n_iter += 1

    @torch.no_grad()
    def _accumulate_scale_step(self) -> None:
        '''scale = E[ (Q_A⊗Q_S)' @ grad)**2 ]'''
        for module in self.modules:
            ekfac_state = self.ekfac_state_dict[module]
            a = self.a_dict[module]
            g = self.g_dict[module]

            # compute per-sample gradient
            if type(module) in (nn.Linear, CustomLinear):
                # a.shape = (B, Nin)
                # g.shape = (B, Nout)
                if module.bias is not None:
                    a = F.pad(a, (0,1), value=1) # (B, Nin+1)
                grad = torch.einsum("bi, bj -> bij", g, a)
            elif type(module) in (nn.Conv2d, CustomConv2d):
                # a.shape = (B, Cin, h, w)
                # g.shape = (B, Cout, h, w)
                a = F.unfold(a, module.kernel_size, module.dilation, module.padding, module.stride) # (B, Cin*k*k, h*w)
                if module.bias is not None:
                    a = F.pad(a, (0,0,0,1,0,0), value=1) # (B, Cin*k*k+1, h*w)
                g = g.reshape(g.size(0), g.size(1), -1) # (B, Cout, h*w)
                grad = torch.einsum("bij, bkj -> bik", g, a)
            elif type(module) in (nn.BatchNorm2d, CustomBatchNorm2d):
                # a.shape = (B, C, h, w)
                # g.shape = (B, C, h, w)
                a = (a - module.running_mean[None, :, None, None]).div(torch.sqrt(module.running_var[None, :, None, None] + module.eps))
                grad_weight = torch.einsum("bchw, bchw -> bc", a, g)
                grad_bias = torch.einsum("bchw -> bc", g)
                grad = torch.cat((torch.diag_embed(grad_weight), grad_bias[:, :, None]), dim=2) # (B, C, C+1)
                # grad = torch.stack((grad_weight, grad_bias), dim=2) # (B, C, 2)
            else:
                raise NotImplementedError
            # grad = torch.chain_matmul(ekfac_state.Q_S.T, grad, ekfac_state.Q_A)
            grad = torch.einsum("ij, bjk, kl -> bil", ekfac_state.Q_S.T, grad, ekfac_state.Q_A)
            ekfac_state.scale.add_(grad.pow(2).mean(0)) # (Nout, Nin) or (Cout, Cin*k*k)
        self.n_iter += 1

    @torch.no_grad()
    def _divide_curvature(self) -> None:
        for module in self.modules:
            kfac_state = self.kfac_state_dict[module]
            kfac_state.A.div_(self.n_iter)
            kfac_state.S.div_(self.n_iter)

    @torch.no_grad()
    def _divide_scale(self) -> None:
        for module in self.modules:
            ekfac_state = self.ekfac_state_dict[module]
            ekfac_state.scale.div_(self.n_iter)

    # @torch.no_grad()
    # def _update_center(self):
    #     for module in self.modules:
    #         if type(module) in (CustomLinear, CustomConv2d, CustomBatchNorm2d):
    #             self.ekfac_state_dict[module].weight = module.weight_tangent.clone()
    #             self.ekfac_state_dict[module].bias = module.bias_tangent.clone() if module.bias_tangent is not None else None
    #         elif type(module) in (nn.Linear, nn.Conv2d, nn.BatchNorm2d):
    #             self.ekfac_state_dict[module].weight = module.weight.clone()
    #             self.ekfac_state_dict[module].bias = module.bias.clone() if module.bias is not None else None
    #         else:
    #             raise NotImplementedError

    def compute_curvature(self, dataset: Dataset, n_steps: int, t: int = None, pseudo_target_fn = torch.normal, batch_size=64) -> None:
        data_loader = MultiEpochsDataLoader(
                            dataset,
                            batch_size=batch_size,
                            shuffle=True,
                            drop_last=True,
                            num_workers=4,
                        )
        data_loader_cycle = icycle(data_loader)

        for m in self.modules:
            for p in m.parameters():
                p.requires_grad_(False)

        hook_handles = self._register_hooks()
        self.model.eval()
        for _ in trange(n_steps, desc="compute curvature"):
            input, _ = next(data_loader_cycle)
            input = input.cuda()
            self.model.zero_grad()
            if t is not None:
                output = self.model(input)[t]
            else:
                output = self.model(input)
            pseudo_target = pseudo_target_fn(output.detach())
            loss = self.criterion(output, pseudo_target).sum(-1).sum()
            loss.backward()
            self._accumulate_curvature_step()

        self._divide_curvature()
        self._remove_hooks(hook_handles)

        self.n_iter = 0

        self._init_ekfac_states()
        # del self.kfac_state_dict

        hook_handles = self._register_hooks()
        self.model.eval()
        for _ in trange(n_steps, desc="compute scaling"):
            input, _ = next(data_loader_cycle)
            input = input.cuda()
            self.model.zero_grad()
            if t is not None:
                output = self.model(input)[t]
            else:
                output = self.model(input)
            pseudo_target = pseudo_target_fn(output.detach())
            loss = self.criterion(output, pseudo_target).sum(-1).sum()
            loss.backward()
            self._accumulate_scale_step()
        self._divide_scale()
        self._remove_hooks(hook_handles)
        self._del_temp_states()

        for m in self.modules:
            for p in m.parameters():
                p.requires_grad_(True)

    def compute_loss(self, center_dict: Mapping[nn.Module, CenterState]) -> torch.Tensor:
        losses = []
        for module in self.modules:
            if type(module) in (CustomLinear, CustomConv2d, CustomBatchNorm2d):
                losses.append(EKFAC_penalty.apply(self.ekfac_state_dict[module], center_dict[module], module.weight_tangent, module.bias_tangent))
            elif type(module) in (nn.Linear, nn.Conv2d, nn.BatchNorm2d):
                losses.append(EKFAC_penalty.apply(self.ekfac_state_dict[module], center_dict[module], module.weight, module.bias))
            else:
                raise NotImplementedError
        loss = sum(losses)
        return loss


class EKFAC_penalty(torch.autograd.Function):
    """
    Hv = (Q_A⊗Q_S) @ Λ @ (Q_A⊗Q_S).T @ v
    out = 0.5 * vHv
    dout/dv = Hv
    """

    @staticmethod
    def forward(ctx, ekfac_state: EKFACState, center_state: CenterState, weight: torch.Tensor, bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        # if weight.ndim == 1: # batchnorm
        #     # weight.shape = (C,)
        #     # bias.shape = (C,)
        #     pass
        weight_aug, fold_weight_fn = unfold_weight(weight, bias)
        weight_aug_center, _ = unfold_weight(center_state.weight, center_state.bias)

        V = weight_aug - weight_aug_center
        # Hv = torch.chain_matmul(
        #     ekfac_state.Q_S,
        #     torch.chain_matmul(ekfac_state.Q_S.T, V, ekfac_state.Q_A) * ekfac_state.scale,
        #     ekfac_state.Q_A.T
        # )

        Hv = torch.einsum("ea, ab, bc, cd, ad, df -> ef", ekfac_state.Q_S, ekfac_state.Q_S.T, V, ekfac_state.Q_A, ekfac_state.scale, ekfac_state.Q_A.T)

        Hdw, Hdb = fold_weight_fn(Hv)
        ctx.save_for_backward(Hdw, Hdb)
        # return torch.dot(V.view(-1), Hv.view(-1)) * 0.5
        return torch.einsum("ab, ab -> ", V, Hv) * 0.5

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        # grad_output.shape == (,)
        Hdw, Hdb = ctx.saved_tensors
        grad_weight = grad_output * Hdw
        if Hdb is not None:
            grad_bias = grad_output * Hdb
        else:
            grad_bias = None
        return None, None, grad_weight, grad_bias
