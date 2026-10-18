"""Optional loss-only causal head; label-free forwards retain full logits."""
import random
import os
from types import MethodType

import torch
import torch.nn.functional as F
import triton.runtime.cache as triton_cache
from transformers.modeling_outputs import CausalLMOutputWithPast


def _head_forward(self, hidden_states, labels=None):
    if labels is None:
        return F.linear(hidden_states, self.weight, self.bias)
    hidden = hidden_states[..., :-1, :].reshape(-1, hidden_states.shape[-1])
    targets = labels[..., 1:].reshape(-1).to(hidden.device)
    indices = torch.nonzero(targets != -100, as_tuple=False).flatten()
    if not hidden.is_cuda or indices.numel() == 0:
        # Preserve the ordinary mean-CE behavior, including all-ignored inputs.
        logits = F.linear(hidden, self.weight, self.bias).float()
        return F.cross_entropy(logits, targets, ignore_index=-100)
    hidden = hidden.index_select(0, indices)
    targets = targets.index_select(0, indices)
    if (self._head_chunk_size > 0 and self.bias is None
            and hidden.dtype == self.weight.dtype == torch.bfloat16):
        from .chunked_head import ChunkedHead
        return ChunkedHead.apply(hidden, self.weight, targets,
                                 self._head_chunk_size, self._head_bf16_accum)
    logits = F.linear(hidden, self.weight, self.bias).float()
    if self._use_liger_ce:
        from liger_kernel.ops.cross_entropy import LigerCrossEntropyFunction
        return LigerCrossEntropyFunction.apply(
            logits, targets, -100, 0.0, 0.0, "mean", None, False)[0]
    return F.cross_entropy(logits, targets)


def enable_fused_head(head):
    # Compilation's temporary filenames must not consume training RNG.
    triton_cache.random = random.SystemRandom()
    head._use_liger_ce = os.environ.get("LLAVA_HEAD_LIGER_CE") == "1"
    head._head_chunk_size = int(os.environ.get("LLAVA_HEAD_CHUNK_SIZE", "0"))
    head._head_bf16_accum = os.environ.get("LLAVA_HEAD_BF16_ACCUM") == "1"
    if head._head_chunk_size < 0:
        raise ValueError("LLAVA_HEAD_CHUNK_SIZE must be nonnegative")
    head.forward = MethodType(_head_forward, head)


def loss_only_forward(model, labels, return_dict=None, **decoder_inputs):
    return_dict = model.config.use_return_dict if return_dict is None else return_dict
    outputs = model.model(return_dict=True, **decoder_inputs)
    # Invoke the existing module so ZeRO parameter gather/backward hooks remain.
    loss = model.lm_head(outputs.last_hidden_state, labels)
    if not return_dict:
        return (loss, None) + outputs.to_tuple()[1:]
    return CausalLMOutputWithPast(
        loss=loss, logits=None, past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states, attentions=outputs.attentions)
