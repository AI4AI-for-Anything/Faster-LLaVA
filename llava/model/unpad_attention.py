"""Project valid rows while retaining the upstream FlashAttention path."""
import torch
from transformers.models.llama import modeling_llama as llama

_dense_forward = llama.LlamaFlashAttention2.forward


def unpadded_projection_forward(self, hidden_states, attention_mask=None,
                                position_ids=None, past_key_value=None,
                                output_attentions=False, use_cache=False, **kwargs):
    if (attention_mask is None or attention_mask.ndim != 2
            or attention_mask.shape != hidden_states.shape[:2]
            or hidden_states.dtype != torch.bfloat16
            or past_key_value is not None or use_cache
            or self.o_proj.bias is not None or 'padding_mask' in kwargs):
        llama.logger.warning_once("Using dense attention projections for an unsupported mask/cache/dtype/bias configuration.")
        return _dense_forward(self, hidden_states, attention_mask, position_ids,
                              past_key_value, output_attentions, use_cache, **kwargs)
    batch, seq, width = hidden_states.shape
    indices = torch.nonzero(attention_mask.reshape(-1), as_tuple=False).flatten()
    active = hidden_states.reshape(-1, width).index_select(0, indices)

    def project(module):
        packed = module(active)
        return packed.new_zeros((batch * seq, packed.shape[-1])).index_copy(
            0, indices, packed).view(batch, seq, -1)

    query = project(self.q_proj).view(batch, seq, self.num_heads, self.head_dim).transpose(1, 2)
    key = project(self.k_proj).view(batch, seq, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    value = project(self.v_proj).view(batch, seq, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    cos, sin = self.rotary_emb(value, seq_len=seq)
    query, key = llama.apply_rotary_pos_emb(query, key, cos, sin, position_ids)
    attention = self._flash_attention_forward(
        query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2),
        attention_mask, seq, dropout=self.attention_dropout if self.training else 0.0)
    packed_output = attention.reshape(batch * seq, self.hidden_size).index_select(0, indices)
    projected = self.o_proj(packed_output)
    output = projected.new_zeros((batch * seq, self.hidden_size)).index_copy(
        0, indices, projected).view(batch, seq, self.hidden_size)
    return output, None, past_key_value


def enable_unpadded_projections():
    llama.LlamaFlashAttention2.forward = unpadded_projection_forward
