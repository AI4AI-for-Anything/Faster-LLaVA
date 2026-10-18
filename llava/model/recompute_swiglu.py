"""Recompute the cheap SwiGLU output instead of retaining it for down backward."""
from types import MethodType

import torch
import torch.nn.functional as F
from liger_kernel.ops.swiglu import swiglu_forward, swiglu_backward


class _SwiGLULinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, gate, up, weight):
        ctx.input_shape = gate.shape
        gate, up, activation = swiglu_forward(gate.contiguous(), up.contiguous())
        output = F.linear(activation, weight)
        ctx.save_for_backward(gate, up, weight)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        gate, up, weight = ctx.saved_tensors
        grad_output = grad_output.contiguous().view(-1, weight.shape[0])
        _, _, activation = swiglu_forward(gate, up)
        grad_weight = grad_output.t().matmul(activation.view(-1, weight.shape[1]))
        del activation
        grad_activation = grad_output.matmul(weight).view_as(gate)
        grad_gate, grad_up = swiglu_backward(gate, up, grad_activation)
        return grad_gate.view(ctx.input_shape), grad_up.view(ctx.input_shape), grad_weight


def _down(self, gate, up=None):
    if up is None:
        return self._swiglu_original_down(gate)
    return _SwiGLULinear.apply(gate, up, self.weight)


def _mlp(self, x):
    if not x.is_cuda or x.dtype != torch.bfloat16 or self.config.pretraining_tp != 1:
        return self._swiglu_original_mlp(x)
    # Keep all original module calls, including the down-projection's ZeRO hooks.
    return self.down_proj(self.gate_proj(x), self.up_proj(x))


def enable_recomputed_swiglu(model):
    for layer in model.layers:
        mlp = layer.mlp
        if mlp.down_proj.bias is not None:
            raise ValueError("Recomputed SwiGLU requires a bias-free down projection")
        mlp._swiglu_original_mlp = mlp.forward
        mlp.forward = MethodType(_mlp, mlp)
        mlp.down_proj._swiglu_original_down = mlp.down_proj.forward
        mlp.down_proj.forward = MethodType(_down, mlp.down_proj)
