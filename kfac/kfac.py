import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torch.utils.hooks import RemovableHandle
from typing import Tuple, List, NamedTuple, Mapping, Optional, Callable, Iterable
from tqdm.auto import tqdm, trange
from dataclasses import dataclass
from collections import OrderedDict
from torch import optim


@dataclass
class KFACState:
    A: torch.Tensor = None
    S: torch.Tensor = None

@dataclass
class CenterState:
    weight: torch.Tensor = None
    bias: Optional[torch.Tensor] = None


def get_center_dict(modules: List[nn.Module]) -> Mapping[nn.Module, CenterState]:
    result = OrderedDict()
    for module in modules:
        if type(module) in (nn.Linear, nn.Conv2d, nn.BatchNorm2d):
            weight = module.weight
            bias = module.bias
        else:
            raise NotImplementedError
        result[module] = CenterState(
            weight=weight.detach().clone(),
            bias=bias.detach().clone() if bias is not None else None
            )
    return result


@torch.no_grad()
def compute_A(module: nn.Module, act: torch.Tensor) -> torch.Tensor:
    r"""
    Computes the outer product of the activations.
    """
    if isinstance(module, nn.Linear):
        # a.shape == (B, Nin)
        b = act.size(0)
        if module.bias is not None: #TODO
            act = F.pad(act, (0,1), value=1) # (B, Nin+1)
        A = torch.einsum("bi, bj -> ij", act, act)
        A /= b
    elif isinstance(module, nn.Conv2d):
        # a.shape == (B, Cin, h, w)
        act = F.unfold(act, module.kernel_size, module.dilation, module.padding, module.stride) # (B, Cin*k*k, h*w)
        bhw = act.size(0)*act.size(2)
        if module.bias is not None:
            act = F.pad(act, (0,0,0,1,0,0), value=1) # (B, Cin*k*k+1, h*w)
        A = torch.einsum("bij, bkj -> ik", act, act) # (Cin*k*k, Cin*k*k)
        A /= bhw
        # A /= a.size(0)
    elif isinstance(module, nn.BatchNorm2d):
        # a.shape == (B, C, h, w)
        b = act.size(0)
        act = (act - module.running_mean[None, :, None, None]).div(torch.sqrt(module.running_var[None, :, None, None] + module.eps))
        if module.bias is not None:
            act = F.pad(act, (0,0,0,0,0,1,0,0), value=1) # (B, C+1, h, w)
        A = torch.einsum("bijk, bljk -> il", act, act) # (C, C)
        A /= b
    else:
        raise NotImplementedError(f'{type(module)}')
    return A


@torch.no_grad()
def compute_S(module: nn.Module, grad: torch.Tensor) -> torch.Tensor:
    r"""
    Computes the outer product of the gradients.
    """
    if isinstance(module, nn.Linear):
        # g.shape == (B, Nout)
        b = grad.size(0)
        S = torch.einsum("bi, bj -> ij", grad, grad)
        S /= b
    elif isinstance(module, nn.Conv2d):
        # g.shape == (B, Cout, h, w)
        b = grad.size(0)
        S = torch.einsum("bijk, bljk -> il", grad, grad)
        S /= b
    elif isinstance(module, nn.BatchNorm2d):
        # g.shape == (B, C, h, w)
        bhw = grad.size(0)*grad.size(2)*grad.size(3)
        S = torch.einsum("bijk, bljk -> il", grad, grad)
        S /= bhw
    else:
        raise NotImplementedError(f'{type(module)}')
    return S


# @torch.no_grad()
def unfold_weight(weight: torch.Tensor, bias: torch.Tensor = None) -> Tuple[torch.Tensor, Callable]:
    """reshapes multidimensional weight tensor to 2D matrix, then augments bias to the weight matrix.

    Args:
        weight (torch.Tensor): weight tensor
        bias (torch.Tensor, optional): bias tensor. Defaults to None.

    Raises:
        ValueError: when weight tensor shape is not supported

    Returns:
        Tuple[torch.Tensor, Callable]: tuple of unfolded-augmented weight matrix and a function to revert the shape.
    """
    is_batchnorm = weight.ndim == 1
    weight_shape = weight.shape
    has_bias = bias is not None

    if weight.ndim == 1:
        weight = weight.diag() # (C, C)
    elif weight.ndim == 2:
        pass # (Nout, Nin)
    elif weight.ndim == 4:
        weight = weight.reshape(weight.size(0), -1) # (Cout, Cin*k*k)
    else:
        raise ValueError(f'{weight.ndim}')

    if has_bias:
        weight_aug = torch.cat((weight, bias[:, None]), dim=1)
    else:
        weight_aug = weight

    def fold_weight_fn(weight_aug: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if has_bias:
            weight, bias = weight_aug[:, :-1], weight_aug[:, -1]
        else:
            weight, bias = weight_aug, None
        if is_batchnorm:
            weight = weight.diagonal() # (C,)
        else:
            weight = weight.reshape(weight_shape)
        return weight, bias

    return weight_aug, fold_weight_fn


class KFAC:
    r"""
    Context manager for computing the Kronecker-factored curvature matrix. 
    The Kronecker-factored curvature matrix is given by

    .. math::
        F = A \otimes S

    where :math:`A` is the outer product of the activations and :math:`S` is the outer product of the gradients.

    Forward pass invokes the update for A matrix, while backward pass invokes the update for S matrix.

    Args:
        layers (List[nn.Module]): List of layers for which to compute the curvature matrix.
        beta (float, optional): Exponential moving average factor. Defaults to 0.99.

    Example:
        >>> layers = [mod for mod in model.modules() if type(mod) in (nn.Linear, nn.Conv2d, nn.BatchNorm2d)]
        >>> with KFAC(layers):
        >>>     output = model(input)
    """

    def __init__(self, layers: List[nn.Module], beta: float = 0.99) -> None:
        assert all([isinstance(m, (nn.Linear, nn.Conv2d, nn.BatchNorm2d)) for m in layers])
        self.state = {layer: KFACState() for layer in layers}  # type: Mapping[nn.Module, KFACState]
        self.beta = beta
        self.hooks = []

    def _forward_hook(self, module, args, output):
        r"""
        Hook for computing the activations.
        """
        input = args[0]
        A_matrix = compute_A(module, input)
        if self.state[module].A is not None:
            self.state[module].A.mul_(self.beta).add_(A_matrix, alpha=1.0-self.beta)
        else:
            self.state[module].A = A_matrix

    def _backward_hook(self, module, grad_output):
        r"""
        Hook for computing the gradients.
        """
        grad_output = grad_output[0]
        S_matrix = compute_S(module, grad_output)
        if self.state[module].S is not None:
            self.state[module].S.mul_(self.beta).add_(S_matrix, alpha=1.0-self.beta)
        else:
            self.state[module].S = S_matrix

    def __enter__(self):
        """Register hooks for computing A and S."""
        for layer in self.state.keys():
            hook_handle = layer.register_forward_pre_hook(self._forward_hook)
            self.hooks.append(hook_handle)
            hook_handle = layer.register_full_backward_pre_hook(self._backward_hook)
            self.hooks.append(hook_handle)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Remove hooks when exiting the context manager."""
        for hook_handle in self.hooks:
            hook_handle.remove()


class KFACRegularizer:

    def __init__(self, model: nn.Module, criterion: nn.Module, modules: List[nn.Module]) -> None:
        self.model = model
        self.criterion = criterion
        self.modules : List[nn.Module] = modules
        self.a_dict : Mapping[nn.Module, torch.Tensor] = OrderedDict()
        self.g_dict : Mapping[nn.Module, torch.Tensor] = OrderedDict()
        self.kfac_state_dict : Mapping[nn.Module, KFACState] = OrderedDict()
        self.n_iter = 0
        self._init_kfac_states()

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
                A=module.weight.new_zeros((a_size, a_size)),
                S=module.weight.new_zeros((g_size, g_size))
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

    def _forward_hook(self, module: nn.Module, input: Tuple[torch.Tensor, ...], output: Tuple[torch.Tensor, ...]) -> None:
        if type(module) in (nn.Linear, nn.Conv2d, nn.BatchNorm2d):
            input, = input
            output = output
        else:
            raise NotImplementedError
        self.a_dict[module] = input.detach().clone()
        def _tensor_backward_hook(grad: torch.Tensor) -> None:
            self.g_dict[module] = grad.detach().clone()
        output.requires_grad_(True).register_hook(_tensor_backward_hook)

    @torch.no_grad()
    def _accumulate_curvature_step(self) -> None:
        for module in self.modules:
            a = self.a_dict[module]
            g = self.g_dict[module]
            kfac_state = self.kfac_state_dict[module]
            A = compute_A(module, a)
            S = compute_S(module, g)
            kfac_state.A.add_(A)
            kfac_state.S.add_(S)
        self.n_iter += 1

    def _divide_curvature(self) -> None:
        for module in self.modules:
            kfac_state = self.kfac_state_dict[module]
            kfac_state.A.div_(self.n_iter)
            kfac_state.S.div_(self.n_iter)

    # @torch.no_grad()
    # def _update_center(self):
    #     for module in self.modules:
    #         if type(module) in (CustomLinear, CustomConv2d, CustomBatchNorm2d):
    #             self.kfac_state_dict[module].weight = module.weight_tangent.clone()
    #             self.kfac_state_dict[module].bias = module.bias_tangent.clone() if module.bias_tangent is not None else None
    #         elif type(module) in (nn.Linear, nn.Conv2d, nn.BatchNorm2d):
    #             self.kfac_state_dict[module].weight = module.weight.clone()
    #             self.kfac_state_dict[module].bias = module.bias.clone() if module.bias is not None else None
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
        self._del_temp_states()

        for m in self.modules:
            for p in m.parameters():
                p.requires_grad_(True)

    def compute_loss(self, center_state_dict: Mapping[nn.Module, CenterState], damping: float = None) -> torch.Tensor:
        loss = 0.
        for module in self.modules:
            kfac_state = self.kfac_state_dict[module]
            center_state = center_state_dict[module]
            if damping is not None:
                dim_a = kfac_state.A.size(0)
                dim_g = kfac_state.S.size(0)
                pi = torch.sqrt((kfac_state.A.trace()*dim_g)/(kfac_state.S.trace()*dim_a))
                damping = pi.new_tensor(damping)
                kfac_state = KFACState(
                    A = kfac_state.A + torch.zeros_like(kfac_state.A).fill_diagonal_(torch.sqrt(damping) * pi),
                    S = kfac_state.S + torch.zeros_like(kfac_state.S).fill_diagonal_(torch.sqrt(damping) / pi)
                )
            if type(module) in (nn.Linear, nn.Conv2d, nn.BatchNorm2d):
                loss += KFAC_penalty.apply(kfac_state, center_state, module.weight, module.bias)
            else:
                raise NotImplementedError
        return loss


def kfac_mvp(A: torch.Tensor, B: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    """Computes the matrix-vector product, where the matrix is factorized by a Kronecker product.
    Uses 'vec trick' to compute the product efficiently.

    Returns:
        torch.Tensor: Matrix-vector product (A⊗B)v
    """
    # (A⊗B)v = vec(BVA')
    m, n = A.shape
    p, q = B.shape
    if vec.ndim == 1:
        return torch.chain_matmul(B, vec.view(q, n), A.T).view(-1)
    elif vec.ndim == 2:
        return torch.chain_matmul(B, vec, A.T) # (p, m)
    else:
        raise ValueError


def kfac_loss_batchnorm(kfac_state: KFACState, center_state: CenterState, weight: torch.Tensor, bias: Optional[torch.Tensor] = None) -> torch.Tensor:
    assert weight.ndim == 1 # is_batchnorm
    S = kfac_state.S # (C, C)
    A = kfac_state.A # (C, C) or (C+1, C+1)

    if bias is not None:
        dw = weight - center_state.weight # (C,)
        db = bias - center_state.bias # (C,)
        Sdw = S * dw[None, :]
        Sdb = torch.mv(S, db)
        Sv = torch.cat((Sdw, Sdb[:, None]), dim=1) # (C, C+1)
        # Hv = Sv @ A # (C, C+1)
        # Hdw = Hv[:, :-1].diagonal() # (C,)
        # Hdb = Hv[:, -1] # (C,)
        Hdw = (Sv * A[:-1, :]).sum(1) # (C,)
        Hdb = torch.mv(Sv, A[:, -1]) # (C,)
        vHv = torch.dot(dw, Hdw) + torch.dot(db, Hdb)
        return 0.5 * vHv, Hdw, Hdb
    else:
        dw = weight - center_state.weight # (C,)
        Sdw = S * dw[None, :]
        Sv = Sdw # (C, C)
        # Hv = Sv @ A # (C, C)
        # Hdw = Hv.diagonal() # (C,)
        Hdw = (Sv * A).sum(1) # (C,)
        vHv = torch.dot(dw, Hdw)
        return 0.5 * vHv, Hdw, None


def kfac_loss_linear_conv(kfac_state: KFACState, center_state: CenterState, weight: torch.Tensor, bias: Optional[torch.Tensor] = None) -> torch.Tensor:
    A = kfac_state.A # (Nin, Nin)
    S = kfac_state.S # (Nout, Nout)
    param, fold_weight_fn = unfold_weight(weight, bias)
    center, fold_weight_fn = unfold_weight(center_state.weight, center_state.bias)
    v = param - center
    Hv = torch.chain_matmul(S, v, A) # (Nout, Nin)
    vHv = torch.dot(v.view(-1), Hv.view(-1))
    Hdw, Hdb = fold_weight_fn(Hv)
    return 0.5 * vHv, Hdw, Hdb


class KFAC_penalty(torch.autograd.Function):
    @staticmethod
    def forward(ctx, kfac_state: KFACState, center_state: CenterState, weight: torch.Tensor, bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        if weight.ndim == 1:
            loss, Hdw, Hdb = kfac_loss_batchnorm(kfac_state, center_state, weight, bias)
        elif weight.ndim == 2 or weight.ndim == 4:
            loss, Hdw, Hdb = kfac_loss_linear_conv(kfac_state, center_state, weight, bias)
        else:
            raise ValueError

        ctx.save_for_backward(Hdw, Hdb)
        return loss

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


class KroneckerBiliearProduct(torch.autograd.Function):
    """Computes the product v.T @ (A⊗B) @ v

    Returns:
        torch.Tensor: bilinear product (A⊗B)v
    """
    # (A⊗B)v = vec(BVA')
    pass




@torch.no_grad()
def compute_trace(module: nn.Module, a: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
    b = a.size(0)
    if isinstance(module, nn.Linear):
        # a.shape == (B, Nin)
        # g.shape == (B, Nout)
        if module.bias is not None: #TODO
            a = F.pad(a, (0,1), value=1) # (B, Nin+1)
        trace = a.square().sum(1) * g.square().sum(1)
        trace = trace.mean()
    elif isinstance(module, nn.Conv2d):
        # a.shape == (B, Cin, h, w)
        # g.shape == (B, Cout, h, w)
        hw = a.size(2) * a.size(3)
        a = F.unfold(a, module.kernel_size, module.dilation, module.padding, module.stride) # (B, Cin*k*k, h*w)
        if module.bias is not None:
            a = F.pad(a, (0,0,0,1,0,0), value=1) # (B, Cin*k*k+1, h*w)
        trace = a.view(b, -1).square().sum(1) * g.view(b, -1).square().sum(1)
        trace = trace.mean().div_(hw)
    else:
        raise NotImplementedError(f'{type(module)}')
    return trace