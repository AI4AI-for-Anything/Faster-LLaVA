"""Retain packed rows through projections, RoPE and varlen attention."""
import torch
from transformers.models.llama import modeling_llama as llama

_dense_forward = llama.LlamaFlashAttention2.forward


def packed_projection_forward(self, hidden_states, attention_mask=None,
                              position_ids=None, past_key_value=None,
                              output_attentions=False, use_cache=False, **kwargs):
    if (attention_mask is None or attention_mask.ndim != 2
            or attention_mask.shape != hidden_states.shape[:2]
            or hidden_states.dtype != torch.bfloat16 or position_ids is None
            or past_key_value is not None or use_cache
            or self.o_proj.bias is not None or 'padding_mask' in kwargs):
        llama.logger.warning_once("Using upstream attention for an unsupported packed-attention configuration.")
        return _dense_forward(self, hidden_states, attention_mask, position_ids,
                              past_key_value, output_attentions, use_cache, **kwargs)
    batch, seq, width = hidden_states.shape
    indices, cu_seqlens, max_seqlen = llama._get_unpad_data(attention_mask)
    active = hidden_states.reshape(-1, width).index_select(0, indices)
    query = self.q_proj(active).view(-1, self.num_heads, self.head_dim)
    key = self.k_proj(active).view(-1, self.num_key_value_heads, self.head_dim)
    value = self.v_proj(active).view(-1, self.num_key_value_heads, self.head_dim)
    positions = position_ids.expand(batch, seq).reshape(-1).index_select(0, indices)
    cos, sin = self.rotary_emb(value, seq_len=seq)
    query, key = llama.apply_rotary_pos_emb(query, key, cos, sin, positions, unsqueeze_dim=1)
    output = llama.flash_attn_varlen_func(
        query, key, value, cu_seqlens_q=cu_seqlens, cu_seqlens_k=cu_seqlens,
        max_seqlen_q=max_seqlen, max_seqlen_k=max_seqlen,
        dropout_p=self.attention_dropout if self.training else 0.0,
        softmax_scale=None, causal=self.is_causal,
        deterministic=torch.are_deterministic_algorithms_enabled())
    projected = self.o_proj(output.reshape(-1, self.hidden_size))
    output = projected.new_zeros((batch * seq, self.hidden_size)).index_copy(
        0, indices, projected).view(batch, seq, self.hidden_size)
    return output, None, past_key_value


def enable_packed_attention():
    llama.LlamaFlashAttention2.forward = packed_projection_forward
