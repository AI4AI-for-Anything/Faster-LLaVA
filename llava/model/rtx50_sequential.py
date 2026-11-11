"""LLaVA-Llama as a RoundPipe ``nn.Sequential`` over packed sequences.

RoundPipe (https://github.com/thustorage/RoundPipe) keeps parameters and
optimizer state in host memory and streams one stage at a time to each GPU, so a
7B model trains on 16 GB cards without data-parallel collectives.  It needs the
model as ``nn.Sequential``; this module provides that view of an existing
``LlavaLlamaForCausalLM`` without copying weights:

    [LlavaPrefix, PackedDecoderLayer x num_layers, PackedHeadLoss]

A microbatch is packed: the expanded token rows of all its samples are
concatenated (no padding) and attention runs per sample through ``cu_seqlens``.
The layout is built on the host by :func:`pack_microbatch`, so the GPU stages do
no host synchronisation.  Layers pass ``(hidden, meta)`` to each other.
"""
from typing import Dict, List, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.llama import modeling_llama as llama

from llava.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX

try:
    from flash_attn import flash_attn_varlen_func
except ImportError:  # PyTorch's own FlashAttention build is used instead.
    flash_attn_varlen_func = None


def pack_microbatch(instances: Sequence[Dict], image_tokens: int, max_length: int,
                    pad_token_id: int, image_dtype: torch.dtype = torch.bfloat16) -> Dict:
    """Pack dataset items into one unpadded microbatch.

    Reproduces ``DataCollatorForSupervisedDataset`` (truncation, the
    ``pad_token_id`` attention mask) and
    ``LlavaMetaForCausalLM.prepare_inputs_labels_for_multimodal`` followed by
    the causal-LM label shift: each image placeholder expands to
    ``image_tokens`` rows labelled ``IGNORE_INDEX``, the expanded sequence is
    truncated to ``max_length``, and the target of a row is the next row's label
    within the same sample.  Images are cast to ``image_dtype`` as the Trainer
    does for DeepSpeed bf16 inputs.
    """
    text_ids, text_rows, image_rows, kept_image_rows = [], [], [], []
    positions, label_rows, label_ids, images = [], [], [], []
    cu, offset, max_seqlen = [0], 0, 0
    for item in instances:
        ids = item['input_ids'][:max_length]
        labels = item['labels'][:max_length]
        attended = ids != pad_token_id
        ids, labels = ids[attended], labels[attended]
        is_image = ids == IMAGE_TOKEN_INDEX
        n_images = int(is_image.sum())
        # Expanded row of every original token: earlier placeholders add image_tokens - 1 rows.
        shift = torch.cumsum(is_image.long(), 0) - is_image.long()
        rows = torch.arange(len(ids)) + shift * (image_tokens - 1)
        length = min(len(ids) + n_images * (image_tokens - 1), max_length)
        full_labels = torch.full((len(ids) + n_images * (image_tokens - 1),), IGNORE_INDEX,
                                 dtype=labels.dtype)
        keep = ~is_image & (rows < length)
        text_ids.append(ids[keep])
        text_rows.append(rows[keep] + offset)
        full_labels[rows[~is_image]] = labels[~is_image]
        for start in rows[is_image].tolist():
            span = torch.arange(start, start + image_tokens)
            ok = span < length
            kept_image_rows.append(torch.arange(image_tokens)[ok] + len(images) * image_tokens)
            image_rows.append(span[ok] + offset)
            images.append(item['image'])
        targets = full_labels[1:length]
        supervised = (targets != IGNORE_INDEX).nonzero().squeeze(1)
        label_rows.append(supervised + offset)
        label_ids.append(targets[supervised])
        positions.append(torch.arange(length))
        offset += length
        cu.append(offset)
        max_seqlen = max(max_seqlen, length)
    cat = lambda xs: torch.cat(xs) if xs else torch.zeros(0, dtype=torch.long)
    return dict(
        input_ids=cat(text_ids), text_rows=cat(text_rows),
        images=torch.stack(images).to(image_dtype) if images else torch.zeros(0, dtype=image_dtype),
        image_rows=cat(image_rows), image_feature_rows=cat(kept_image_rows),
        position_ids=cat(positions), label_rows=cat(label_rows), label_ids=cat(label_ids),
        cu_seqlens=torch.tensor(cu, dtype=torch.int32), max_seqlen=max_seqlen,
        num_rows=offset,
    )


def split_balanced(lengths: Sequence[int], num_microbatch: int) -> List[List[int]]:
    """Greedy longest-first split of sample indices into token-balanced groups."""
    groups = [[] for _ in range(num_microbatch)]
    loads = [0] * num_microbatch
    for i in sorted(range(len(lengths)), key=lambda i: -lengths[i]):
        j = min(range(num_microbatch), key=loads.__getitem__)
        groups[j].append(i)
        loads[j] += lengths[i]
    return [sorted(g) for g in groups if g]


class LlavaPrefix(nn.Module):
    """CLIP (frozen), projector and token embedding into the packed rows."""

    def __init__(self, model):
        super().__init__()
        self.embed_tokens = model.get_model().embed_tokens
        self.vision_tower = model.get_vision_tower()
        self.mm_projector = model.get_model().mm_projector
        self.hidden_size = model.config.hidden_size

    def forward(self, batch):
        from roundpipe import doing_recompute, get_recompute_data, save_for_recompute

        images = batch['images']
        if images.shape[0] == 0:
            features = None
        elif doing_recompute():
            features, = get_recompute_data()  # CLIP is frozen: reuse, do not recompute.
        else:
            features = self.vision_tower(images.to(self.vision_tower.dtype))
            save_for_recompute(features)
        tokens = self.embed_tokens(batch['input_ids'])
        hidden = tokens.new_zeros(batch['num_rows'], self.hidden_size)
        hidden = hidden.index_copy(0, batch['text_rows'], tokens)
        if features is not None:
            projected = self.mm_projector(features).flatten(0, 1)
            projected = projected.index_select(0, batch['image_feature_rows'])
            hidden = hidden.index_copy(0, batch['image_rows'], projected)
        meta = (batch['cu_seqlens'], batch['max_seqlen'], batch['position_ids'],
                batch['label_rows'], batch['label_ids'])
        return hidden, meta


def _varlen_attention(q, k, v, cu_seqlens, max_seqlen):
    if flash_attn_varlen_func is not None:
        return flash_attn_varlen_func(
            q, k, v, cu_seqlens_q=cu_seqlens, cu_seqlens_k=cu_seqlens,
            max_seqlen_q=max_seqlen, max_seqlen_k=max_seqlen, dropout_p=0.0,
            causal=True, deterministic=torch.are_deterministic_algorithms_enabled())
    from torch.nn.attention.varlen import varlen_attn
    return varlen_attn(q, k, v, cu_seqlens, cu_seqlens, max_seqlen, max_seqlen,
                       window_size=(-1, 0))


class PackedDecoderLayer(nn.Module):
    """One ``LlamaDecoderLayer`` on packed rows (same math as ``packed_decoder.py``)."""

    def __init__(self, layer, rope_extent: int):
        super().__init__()
        self.layer = layer
        self.rope_extent = rope_extent

    def forward(self, state):
        hidden, meta = state
        cu_seqlens, max_seqlen, positions = meta[:3]
        layer, attn = self.layer, self.layer.self_attn
        x = layer.input_layernorm(hidden)
        q = attn.q_proj(x).view(-1, attn.num_heads, attn.head_dim)
        k = attn.k_proj(x).view(-1, attn.num_key_value_heads, attn.head_dim)
        v = attn.v_proj(x).view(-1, attn.num_key_value_heads, attn.head_dim)
        cos, sin = attn.rotary_emb(v, seq_len=self.rope_extent)
        q, k = llama.apply_rotary_pos_emb(q, k, cos, sin, positions, unsqueeze_dim=1)
        x = _varlen_attention(q, k, v, cu_seqlens, max_seqlen)
        hidden = hidden + attn.o_proj(x.reshape(-1, attn.hidden_size))
        hidden = hidden + layer.mlp(layer.post_attention_layernorm(hidden))
        return hidden, meta


class PackedHeadLoss(nn.Module):
    """Final norm, LM head and summed cross-entropy over supervised rows only."""

    def __init__(self, model):
        super().__init__()
        self.norm = model.get_model().norm
        self.lm_head = model.lm_head

    def forward(self, state):
        hidden, meta = state
        label_rows, label_ids = meta[3], meta[4]
        logits = self.lm_head(self.norm(hidden.index_select(0, label_rows))).float()
        return F.cross_entropy(logits, label_ids, reduction='sum')


def to_sequential(model) -> nn.Sequential:
    """View ``LlavaLlamaForCausalLM`` as RoundPipe layers; parameters are shared, not copied."""
    if model.config.pretraining_tp != 1:
        raise ValueError('RTX 50 recipe requires pretraining_tp == 1')
    if getattr(model.config, 'mm_patch_merge_type', 'flat') != 'flat':
        raise ValueError('RTX 50 recipe supports the LLaVA-1.5 flat image features only')
    # Cover every position of a truncated sample, as the padded upstream path does.
    rope_extent = max(model.config.max_position_embeddings, getattr(model.config, 'tokenizer_model_max_length', 0) or 0)
    layers = [LlavaPrefix(model)]
    layers += [PackedDecoderLayer(layer, rope_extent) for layer in model.get_model().layers]
    layers.append(PackedHeadLoss(model))
    return nn.Sequential(*layers)
