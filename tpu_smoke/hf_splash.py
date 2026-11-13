"""Pallas splash attention behind HuggingFace's attention interface (LLaVA's LM).

HF transformers 4.56 dispatches LlamaAttention through
`ALL_ATTENTION_FUNCTIONS[config._attn_implementation]`. Registering a new key
lets the upstream LLaVA model use splash without editing its files: set
`config._attn_implementation = "xla_splash"` before building the model.

Supported case: causal self-attention with no padding (the LLaVA smoke batch is
fixed length, attention_mask all ones). HF may still hand over a 4-D additive
mask; when it does, it MUST be the plain causal mask, which this checks once on
the host -- a padded batch would need segment ids and is refused rather than
silently attended across.

Numerical tier N1 (online softmax). Equivalence vs HF sdpa is checked in
tpu_smoke/test_hf_splash_equiv.py before timing.
"""
import os

import torch

from tpu_smoke import splash_config as _ns    # pins Mosaic IR version; mesh state
from torch_xla.experimental.splash_attention import splash_attention

_CHECKED = {"mask": False}


def xla_splash_attention_forward(module, query, key, value, attention_mask, scaling=None,
                                 dropout=0.0, **kwargs):
    """HF signature: q [B, H, T, D], k/v [B, Hkv, T, D] -> ([B, T, H, D], None)."""
    if dropout:
        raise NotImplementedError("splash HF shim: dropout")
    B, H, T, D = query.shape
    if attention_mask is not None and not _CHECKED["mask"]:
        m = attention_mask[..., :T, :T].float().cpu()
        causal = torch.triu(torch.full((T, T), float("-inf")), 1)
        ok = torch.equal(torch.isinf(m).expand(B, -1, T, T)[:, 0], torch.isinf(causal).expand(B, T, T))
        if not ok:
            raise NotImplementedError("splash HF shim: mask is not plain causal (padding?)")
        _CHECKED["mask"] = True
    if key.shape[1] != H:
        key = key.repeat_interleave(H // key.shape[1], dim=1)
        value = value.repeat_interleave(H // value.shape[1], dim=1)
    scale = D ** -0.5 if scaling is None else scaling
    # Splash blocks must divide the sequence (LLaVA: 2047 text + 576 image =
    # 2623, 11-06 "q_block_size=512 should divide q_seq_len=2623"). Pad at the
    # END: under the causal mask a real query at position t < T only sees keys
    # <= t, so trailing pad keys are never attended and the real rows are the
    # same function; the padded query rows are sliced off before returning.
    # Block size matters more than padding: 2623 padded to 2688 forced 128-wide
    # blocks and made LLaVA 8.6% SLOWER (11-06). SPLASH_PAD_TO=512 pads to a
    # multiple of 512 so 512-wide blocks fit, at the cost of the padded rows.
    Tp = -(-T // int(os.environ.get("SPLASH_PAD_TO", "128"))) * int(os.environ.get("SPLASH_PAD_TO", "128"))
    q = query * scale
    if Tp != T:
        pad = (0, 0, 0, Tp - T)
        q, key, value = (torch.nn.functional.pad(t, pad) for t in (q, key, value))
    blk = 512 if Tp % 512 == 0 else (256 if Tp % 256 == 0 else 128)
    cfg = _ns._config(blk, None, _ns._MESH["name"], _ns._MESH["spec"])
    out = splash_attention(q, key, value, cfg, None, True)[:, :, :T]
    return out.transpose(1, 2).contiguous(), None


def register():
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    ALL_ATTENTION_FUNCTIONS["xla_splash"] = xla_splash_attention_forward
