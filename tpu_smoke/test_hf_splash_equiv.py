"""TPU check: hf_splash.xla_splash_attention_forward == HF sdpa_attention_forward.

Locks the property the LLaVA splash path must have before timing: on the shapes
the Vicuna-7B LM trains with (H=32, D=128, T=2623 by default, env T overrides, bf16), causal, with the 4-D
additive causal mask HF builds, the splash forward output and q/k/v gradients
agree with HF's own sdpa attention within a bf16 tolerance. Also checks that a
padded (non-causal) mask is REFUSED rather than silently ignored.

Run on one host (4 chips, SPMD), torch_xla with the pallas extra:
  PJRT_DEVICE=TPU python -m tpu_smoke.test_hf_splash_equiv
"""
import os
import sys
import types

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))

import torch_xla
import torch_xla.runtime as xr
import torch_xla.distributed.spmd as xs

xr.use_spmd()

from transformers.integrations.sdpa_attention import sdpa_attention_forward  # noqa: E402
from tpu_smoke import hf_splash, splash_config                              # noqa: E402


def rel(a, b):
    return float((a - b).abs().max() / b.abs().max().clamp_min(1e-6))


def run(fn, q, k, v, mask, g):
    q, k, v = (t.detach().clone().requires_grad_(True) for t in (q, k, v))
    mod = types.SimpleNamespace(is_causal=True, num_key_value_groups=1, training=False)
    o, _ = fn(mod, q, k, v, mask, scaling=q.shape[-1] ** -0.5, dropout=0.0)
    (o.float() * g).sum().backward()
    torch_xla.sync()
    return [t.cpu().float() for t in (o, q.grad, k.grad, v.grad)]


def main():
    n = xr.global_runtime_device_count()
    mesh = xs.Mesh(np.arange(n), (n, 1), ("data", "fsdp"))
    splash_config.set_mesh(mesh, "data")
    dev = torch_xla.device()
    B, H, T, D = n, 32, int(os.environ.get("T", "2623")), 128   # LLaVA: 2047 text + 576 image
    gen = torch.Generator().manual_seed(0)
    q, k, v = (torch.randn(B, H, T, D, generator=gen).to(torch.bfloat16).to(dev) for _ in range(3))
    g = torch.randn(B, T, H, D, generator=gen).to(dev)
    for t in (q, k, v):
        xs.mark_sharding(t, mesh, (("data", "fsdp"), None, None, None))
    xs.mark_sharding(g, mesh, (("data", "fsdp"), None, None, None))
    causal = torch.triu(torch.full((T, T), float("-inf")), 1).to(torch.bfloat16)
    mask = causal[None, None].expand(B, 1, T, T).contiguous().to(dev)
    ref = run(sdpa_attention_forward, q, k, v, mask, g)
    got = run(hf_splash.xla_splash_attention_forward, q, k, v, mask, g)
    errs = [rel(a, b) for a, b in zip(got, ref)]
    print("splash vs HF sdpa: " + " ".join(f"{nm}={e:.2e}" for nm, e in zip(("out", "dq", "dk", "dv"), errs)))
    assert max(errs) < 2e-2, errs
    # A padded mask (last 100 keys masked for every query) must be refused.
    hf_splash._CHECKED["mask"] = False
    padded = causal.clone(); padded[:, -100:] = float("-inf")
    try:
        run(hf_splash.xla_splash_attention_forward, q, k, v,
            padded[None, None].expand(B, 1, T, T).contiguous().to(dev), g)
    except NotImplementedError:
        print("PASS padded mask refused")
    else:
        raise AssertionError("padded mask accepted")
    print("PASS hf splash == HF sdpa (fwd + grads)")


if __name__ == "__main__":
    main()
