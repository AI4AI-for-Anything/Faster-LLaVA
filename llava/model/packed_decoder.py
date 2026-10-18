"""Keep right-padded decoder inputs packed across every layer."""
from dataclasses import dataclass
from types import MethodType
import os
import torch
from transformers.models.llama import modeling_llama as llama
from transformers.modeling_outputs import BaseModelOutputWithPast


@dataclass(frozen=True)
class PackedLayout:
    cu: torch.Tensor
    max_seq: int
    rope_extent: int


def _attention(self, hidden_states, attention_mask=None, position_ids=None,
               past_key_value=None, output_attentions=False, use_cache=False, **kwargs):
    if not isinstance(attention_mask, PackedLayout):
        return self._packed_original_attention(hidden_states, attention_mask, position_ids,
            past_key_value, output_attentions, use_cache, **kwargs)
    layout = attention_mask
    q = self.q_proj(hidden_states).view(-1, self.num_heads, self.head_dim)
    k = self.k_proj(hidden_states).view(-1, self.num_key_value_heads, self.head_dim)
    v = self.v_proj(hidden_states).view(-1, self.num_key_value_heads, self.head_dim)
    cos, sin = self.rotary_emb(v, seq_len=layout.rope_extent)
    if self._packed_liger_rope:
        from .packed_rope import packed_liger_rope
        q, k = packed_liger_rope(q, k, cos, sin, position_ids)
    else:
        q, k = llama.apply_rotary_pos_emb(q, k, cos, sin, position_ids, unsqueeze_dim=1)
    x = llama.flash_attn_varlen_func(q, k, v,
        cu_seqlens_q=layout.cu, cu_seqlens_k=layout.cu,
        max_seqlen_q=layout.max_seq, max_seqlen_k=layout.max_seq,
        dropout_p=self.attention_dropout if self.training else 0.0,
        softmax_scale=None, causal=self.is_causal,
        deterministic=torch.are_deterministic_algorithms_enabled())
    return self.o_proj(x.reshape(-1, self.hidden_size)), None, None


def _layer(self, hidden_states, attention_mask=None, position_ids=None,
           past_key_value=None, output_attentions=False, use_cache=False, **kwargs):
    if not isinstance(attention_mask, PackedLayout):
        return self._packed_original_layer(hidden_states, attention_mask, position_ids,
            past_key_value, output_attentions, use_cache, **kwargs)
    residual = hidden_states
    x = self.input_layernorm(hidden_states)
    x = self.self_attn(x, attention_mask=attention_mask, position_ids=position_ids,
        output_attentions=False, use_cache=False)[0]
    x = residual + x
    return (x + self.mlp(self.post_attention_layernorm(x)),)


def _model(self, input_ids=None, attention_mask=None, position_ids=None,
           past_key_values=None, inputs_embeds=None, use_cache=None,
           output_attentions=None, output_hidden_states=None, return_dict=None):
    cache = self.config.use_cache if use_cache is None else use_cache
    attns = self.config.output_attentions if output_attentions is None else output_attentions
    states = self.config.output_hidden_states if output_hidden_states is None else output_hidden_states
    kwargs = dict(input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids,
        past_key_values=past_key_values, inputs_embeds=inputs_embeds, use_cache=use_cache,
        output_attentions=output_attentions, output_hidden_states=output_hidden_states,
        return_dict=return_dict)
    if (cache or attns or states or past_key_values is not None
            or not self._use_flash_attention_2 or attention_mask is None
            or attention_mask.ndim != 2
            or (input_ids is not None and inputs_embeds is not None)):
        return self._packed_original_model(**kwargs)
    mask = attention_mask.bool()
    # Only valid prefixes: a left-pad position may predict a valid next label.
    if bool((mask[:, 1:] & ~mask[:, :-1]).any()) or bool((mask.sum(-1) == 0).any()):
        return self._packed_original_model(**kwargs)
    if inputs_embeds is None:
        if input_ids is None:
            return self._packed_original_model(**kwargs)
        inputs_embeds = self.embed_tokens(input_ids)
    if inputs_embeds.dtype != torch.bfloat16 or inputs_embeds.shape[:2] != mask.shape:
        return self._packed_original_model(**kwargs)
    batch, seq, width = inputs_embeds.shape
    indices, cu, max_seq = llama._get_unpad_data(attention_mask)
    x = inputs_embeds.reshape(-1, width).index_select(0, indices)
    if position_ids is None:
        position_ids = torch.arange(seq, device=x.device).unsqueeze(0)
    positions = position_ids.expand(batch, seq).reshape(-1).index_select(0, indices)
    layout = PackedLayout(cu, max_seq, seq)
    for layer in self.layers:
        if self.gradient_checkpointing and self.training:
            outputs = self._gradient_checkpointing_func(
                layer.__call__, x, layout, positions, None, False, False)
        else:
            outputs = layer(x, attention_mask=layout, position_ids=positions,
                output_attentions=False, use_cache=False)
        x = outputs[0]
    x = self.norm(x)
    x = x.new_zeros((batch * seq, width)).index_copy(0, indices, x).view(batch, seq, width)
    if not (self.config.use_return_dict if return_dict is None else return_dict):
        return (x,)
    return BaseModelOutputWithPast(last_hidden_state=x)


def enable_packed_decoder(model):
    if hasattr(model, '_packed_original_model'):
        return
    model._packed_original_model = model.forward
    model.forward = MethodType(_model, model)
    for layer in model.layers:
        layer._packed_original_layer = layer.forward
        layer.forward = MethodType(_layer, layer)
        attn = layer.self_attn
        attn._packed_liger_rope = os.environ.get("LLAVA_PACKED_LIGER_ROPE") == "1"
        attn._packed_original_attention = attn.forward
        attn.forward = MethodType(_attention, attn)
