"""Avoid token-local MLP work on masked rows; keep attention layout unchanged."""

import torch
from transformers.models.llama.modeling_llama import LlamaDecoderLayer


_dense_forward = LlamaDecoderLayer.forward


def unpadded_mlp_forward(self, hidden_states, attention_mask=None,
                         position_ids=None, past_key_value=None,
                         output_attentions=False, use_cache=False, **kwargs):
    if (attention_mask is None or attention_mask.ndim != 2
            or attention_mask.shape != hidden_states.shape[:2]
            or past_key_value is not None or use_cache
            or "padding_mask" in kwargs):
        return _dense_forward(
            self, hidden_states, attention_mask, position_ids, past_key_value,
            output_attentions, use_cache, **kwargs)

    residual = hidden_states
    hidden_states = self.input_layernorm(hidden_states)
    hidden_states, attn_weights, present = self.self_attn(
        hidden_states=hidden_states, attention_mask=attention_mask,
        position_ids=position_ids, past_key_value=past_key_value,
        output_attentions=output_attentions, use_cache=use_cache, **kwargs)
    hidden_states = residual + hidden_states

    # Also retain the position before each valid token: its logits can predict
    # that token under shifted causal loss (including left-padded inputs).
    needed = attention_mask.bool().clone()
    needed[:, :-1] |= attention_mask[:, 1:].bool()
    indices = torch.nonzero(needed.reshape(-1), as_tuple=False).flatten()
    flat = hidden_states.reshape(-1, hidden_states.shape[-1])
    active = flat.index_select(0, indices)
    update = self.mlp(self.post_attention_layernorm(active))
    hidden_states = flat.index_add(0, indices, update).view_as(hidden_states)
    outputs = (hidden_states,)
    if output_attentions:
        outputs += (attn_weights,)
    if use_cache:
        outputs += (present,)
    return outputs


def enable_unpadded_mlp():
    LlamaDecoderLayer.forward = unpadded_mlp_forward
