"""Bounded-memory loss-only head using native GEMMs and FP32 cross entropy."""
import torch
import torch.nn.functional as F
from liger_kernel.ops.cross_entropy import cross_entropy_forward


def _ce_sum(logits, targets):
    loss, _, dlogits = cross_entropy_forward(
        logits, targets, -100, 0.0, 0.0, 'sum', None, False)
    return loss, dlogits


def _gemm_rows(hidden):
    # The installed BF16 GEMV dispatch faults on very small output-head row counts.
    # Keep tiny tails on the GEMM path; padded rows never enter the loss.
    if hidden.shape[0] < 32:
        return F.pad(hidden, (0, 0, 0, 32 - hidden.shape[0]))
    return hidden


class ChunkedHead(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hidden, weight, targets, chunk_size, bf16_accum=False):
        # Caller has already selected every valid shifted label exactly once.
        count = targets.numel()
        if count == 0:
            raise ValueError('ChunkedHead requires at least one valid label')
        ctx.save_for_backward(hidden, weight, targets)
        ctx.chunk_size = chunk_size
        ctx.bf16_accum = bf16_accum
        loss = torch.zeros((), device=hidden.device, dtype=torch.float32)
        for start in range(0, count, chunk_size):
            end = min(start + chunk_size, count)
            block = _gemm_rows(hidden[start:end])
            logits = F.linear(block, weight)[:end-start].float()
            part, scratch_grad = _ce_sum(logits, targets[start:end])
            loss.add_(part)
            del block, logits, part, scratch_grad
        return loss / count

    @staticmethod
    def backward(ctx, grad_output):
        hidden, weight, targets = ctx.saved_tensors
        count = targets.numel()
        grad_hidden = torch.empty_like(hidden)
        grad_weight = torch.zeros(weight.shape, device=weight.device,
                                  dtype=weight.dtype if ctx.bf16_accum else torch.float32)
        # Apply the actual upstream scale before the BF16 GEMM, including Trainer
        # global-token normalization and accumulation factors.
        scale = grad_output.float() / count
        for start in range(0, count, ctx.chunk_size):
            end = min(start + ctx.chunk_size, count)
            block = _gemm_rows(hidden[start:end])
            logits = F.linear(block, weight)[:end-start].float()
            _, dlogits = _ce_sum(logits, targets[start:end])
            dlogits.mul_(scale)
            dg = _gemm_rows(dlogits.to(hidden.dtype))
            if block.shape[0] == end - start:
                torch.mm(dg, weight, out=grad_hidden[start:end])
            else:
                grad_hidden[start:end].copy_(dg.matmul(weight)[:end-start])
            if ctx.bf16_accum:
                # Accumulate directly into the original gradient dtype, avoiding
                # a full vocabulary-by-hidden temporary for every token chunk.
                grad_weight.addmm_(dg.t(), block)
            else:
                grad_weight.add_(dg.t().matmul(block))
            del block, logits, dlogits, dg
        result = (grad_hidden, grad_weight.to(weight.dtype), None, None, None)
        return result[:len(ctx.needs_input_grad)]
