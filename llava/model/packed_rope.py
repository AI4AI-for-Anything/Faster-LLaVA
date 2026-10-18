"""Adapt existing Liger Q/K rotary kernels to packed decoder rows."""
from liger_kernel.ops.rope import LigerRopeFunction


def packed_liger_rope(q, k, cos, sin, positions):
    # Position lookup is unchanged; the synthetic batch dimension is only a view.
    cos = cos.index_select(0, positions).unsqueeze(0)
    sin = sin.index_select(0, positions).unsqueeze(0)
    q, k = LigerRopeFunction.apply(
        q.transpose(0, 1).unsqueeze(0), k.transpose(0, 1).unsqueeze(0),
        cos, sin)
    return q.squeeze(0).transpose(0, 1), k.squeeze(0).transpose(0, 1)
