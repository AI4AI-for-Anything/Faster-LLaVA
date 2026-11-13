"""Static-shape replacement for LLaVA's image splice, for XLA.

Upstream `LlavaMetaForCausalLM.prepare_inputs_labels_for_multimodal`
(llava/model/llava_arch.py:145) builds every sample in a Python
loop: `torch.where(ids == IMAGE_TOKEN_INDEX)[0].tolist()` pulls data to the host
(a graph break on XLA), masked indexing by attention_mask gives data-dependent
shapes, and the padded batch is re-stacked from per-sample pieces with
`torch.full`/`torch.zeros` fills. On a 64-chip SPMD mesh (11-06) that path
either replicated the whole global batch on every chip (57.8 GB score tensor) or,
once its outputs were re-annotated, aborted the SPMD partitioner
(`spmd_partitioner_util.h:127 Check failed: IsScalarWithElementType`).

This computes the SAME outputs for the case the LLaVA-1.5 finetune recipe
produces -- exactly ONE <image> token per sample, right padding
(tokenizer_padding_side 'right'), no tokenizer_model_max_length truncation
needed (text_len - 1 + n_img <= model_max_length) -- with batched static ops:

  out[b, t] = text_embed[b, t]            for t <  p_b
              image_feat[b, t - p_b]      for p_b <= t < p_b + N
              text_embed[b, t - N + 1]    for t >= p_b + N

where p_b is the image token position and N the number of image patches. Upstream
drops attention-masked tokens before splicing and re-pads on the right, so
padding positions must sit at the END of each row for this to be equal; that is
asserted on the host from the attention mask, never assumed.

Equivalence with upstream is tested bitwise on CPU in
tpu_smoke/test_llava_static_splice.py. Anything outside the supported case
raises instead of silently diverging.
"""
import torch

from llava.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX


def static_splice(model, input_ids, attention_mask, labels, images, image_pos, uniform_pos=None):
    """Returns (position_ids, attention_mask, inputs_embeds, labels) as upstream.

    image_pos: LongTensor [B], the index of the single <image> token per row.
    It is computed by the caller (on the host, from the data) because finding it
    on device is exactly the data-dependent step this replaces.
    """
    B, T = input_ids.shape
    feats = model.encode_images(images)                      # [B, N, h]
    N = feats.shape[1]
    L = T - 1 + N
    dev = input_ids.device
    if uniform_pos is not None:
        return _splice_at(model, input_ids, attention_mask, labels, feats, int(uniform_pos))

    safe_ids = torch.where(input_ids == IMAGE_TOKEN_INDEX, torch.zeros_like(input_ids), input_ids)
    text = model.get_model().embed_tokens(safe_ids)          # [B, T, h]

    t = torch.arange(L, device=dev).unsqueeze(0)             # [1, L]
    p = image_pos.to(dev).unsqueeze(1)                       # [B, 1]
    in_img = (t >= p) & (t < p + N)
    after = t >= p + N
    text_idx = torch.where(after, t - N + 1, t).clamp(max=T - 1)
    img_idx = (t - p).clamp(0, N - 1)

    text_g = torch.gather(text, 1, text_idx.unsqueeze(-1).expand(B, L, text.shape[-1]))
    img_g = torch.gather(feats, 1, img_idx.unsqueeze(-1).expand(B, L, feats.shape[-1]))
    embeds = torch.where(in_img.unsqueeze(-1), img_g.to(text_g.dtype), text_g)

    lab_text = torch.gather(labels, 1, text_idx)
    new_labels = torch.where(in_img, torch.full_like(lab_text, IGNORE_INDEX), lab_text)

    am_text = torch.gather(attention_mask.long(), 1, text_idx).bool()
    new_mask = in_img | am_text
    # Upstream zeroes padded embeddings and sets their labels to IGNORE_INDEX.
    embeds = embeds * new_mask.unsqueeze(-1).to(embeds.dtype)
    new_labels = torch.where(new_mask, new_labels, torch.full_like(new_labels, IGNORE_INDEX))
    # Upstream: arange over the real length, zeros on the padding.
    position_ids = torch.where(new_mask, t.expand(B, L), torch.zeros_like(t.expand(B, L)))
    return position_ids, new_mask.to(attention_mask.dtype), embeds, new_labels


def _splice_at(model, input_ids, attention_mask, labels, feats, p):
    """static_splice when EVERY row has its <image> at the same host-known position p.

    The general path gathers along dim 1 with per-row indexes; on TPU XLA expands
    that gather into a dense compare-select-reduce over all L x T position pairs
    (11-06 LLaVA profile: fusion.31/30/32 + select_reduce_fusion, ~220 ms of a
    1300 ms device step, 17%). With one shared p the splice is static slicing and
    concatenation, O(L*h) copies, same values (test_llava_static_splice.py).
    """
    B, T = input_ids.shape
    N = feats.shape[1]
    L = T - 1 + N
    dev = input_ids.device
    safe_ids = torch.where(input_ids == IMAGE_TOKEN_INDEX, torch.zeros_like(input_ids), input_ids)
    text = model.get_model().embed_tokens(safe_ids)          # [B, T, h]
    embeds = torch.cat([text[:, :p], feats.to(text.dtype), text[:, p + 1:]], 1)
    am = attention_mask.bool()
    new_mask = torch.cat([am[:, :p], torch.ones(B, N, dtype=torch.bool, device=dev), am[:, p + 1:]], 1)
    lab = torch.cat([labels[:, :p], torch.full((B, N), IGNORE_INDEX, dtype=labels.dtype, device=dev),
                     labels[:, p + 1:]], 1)
    embeds = embeds * new_mask.unsqueeze(-1).to(embeds.dtype)
    new_labels = torch.where(new_mask, lab, torch.full_like(lab, IGNORE_INDEX))
    t = torch.arange(L, device=dev).unsqueeze(0).expand(B, L)
    position_ids = torch.where(new_mask, t, torch.zeros_like(t))
    return position_ids, new_mask.to(attention_mask.dtype), embeds, new_labels


def uniform_image_pos(image_pos):
    """Host side: the shared <image> position if every row has the same one, else None."""
    return int(image_pos[0]) if bool((image_pos == image_pos[0]).all()) else None


def check_supported(input_ids, attention_mask):
    """Host-side guard. Raises unless the batch is the case static_splice covers."""
    n_img = (input_ids == IMAGE_TOKEN_INDEX).sum(1)
    if not bool((n_img == 1).all()):
        raise ValueError(f"static_splice needs exactly one <image> per sample, got {n_img.tolist()}")
    am = attention_mask.bool()
    lens = am.sum(1)
    right_padded = torch.arange(am.shape[1]).unsqueeze(0) < lens.unsqueeze(1)
    if not bool((right_padded == am).all()):
        raise ValueError("static_splice needs right padding (mask = 1...1 0...0 per row)")
    pos = (input_ids == IMAGE_TOKEN_INDEX).int().argmax(1)
    if not bool((pos < lens).all()):
        raise ValueError("<image> token falls in the padding")
    return pos
