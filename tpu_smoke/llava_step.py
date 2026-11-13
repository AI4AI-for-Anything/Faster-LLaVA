"""Does upstream LLaVA's model code build and step on TPU (torch_xla)?

Faster-LLaVA pins torch 2.1.2 / transformers 4.37.2, which cannot share a venv
with torch_xla 2.9; this imports its `llava` package unchanged under the TPU
venv's transformers 4.56.2 and finds out where it breaks. The recipe's opt-in
CUDA switches (scripts/configs/faster_7b.sh: Apex RMSNorm, Liger, packed
attention, ...) are all left unset, i.e. upstream LLaVA-1.5 behaviour.

Model: LlavaLlamaForCausalLM with the 7B recipe's multimodal wiring
(mlp2x_gelu projector, vision select layer -2, CLIP ViT-L/14-336 vision tower,
image_aspect_ratio pad). The language model is RANDOMLY initialised from the
Vicuna-7B config, optionally shrunk with --layers to fit a quick smoke; the
vision tower is the real pretrained CLIP. So this checks code paths and step
time, not quality.

Batch: synthetic. Each sample is a prompt with one <image> token followed by
text, with the label masked on the prompt as in LLaVA's preprocess. Fixed
length, so the graph is static.
"""
import argparse
import json
import os
import statistics
import sys
import time
import types

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)          # repository root: holds llava/ and tpu_smoke/
sys.path.insert(0, _ROOT)

import torch_xla
import torch_xla.debug.metrics as tmetrics
from torch_xla.core import xla_model as xm


def uncached():
    names = tmetrics.counter_names()
    return tmetrics.counter_value("UncachedCompile") if "UncachedCompile" in names else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=4, help="GLOBAL batch")
    ap.add_argument("--seq", type=int, default=1024, help="text tokens incl. the image token")
    ap.add_argument("--layers", type=int, default=4, help="LM layers (Vicuna-7B has 32)")
    ap.add_argument("--warmup", type=int, default=6)
    ap.add_argument("--window", type=int, default=10)
    ap.add_argument("--windows", type=int, default=4)
    ap.add_argument("--spmd", action="store_true")
    ap.add_argument("--splice-concat", action="store_true",
                    help="static_splice fast path for a batch-shared <image> position: "
                         "slice+concat instead of a gather XLA expands densely (bitwise)")
    ap.add_argument("--weight-grad-hint", action="store_true",
                    help="trainable Linear/Embedding forwards use xs.mark_sharding_with_gradients(weight): "
                         "the weight GRADIENT carries the param's sharding where it is produced, unlike "
                         "--shard-grads (annotation after backward, ignored by XLA, 11-07). Layout only")
    ap.add_argument("--shard-grads", action="store_true",
                    help="annotate each param grad with its param's sharding before use: the 1536 "
                         "profile showed full-weight grad all-reduces (21%% of device time) where "
                         "FSDP needs a reduce-scatter (11-07)")
    ap.add_argument("--traj", type=int, default=0,
                    help="statistical-gate mode: N steps from init, fresh batch per step from (seed, "
                         "step): text walks a FIXED sparse Markov chain (4 successors per token, "
                         "shared by all seeds) so the LM has something to learn; prints TRAJ json")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lr-scale", type=float, default=1.0, help="sensitivity control")
    ap.add_argument("--profile", default="", help="dir for an xplane trace of 3 steady steps")
    ap.add_argument("--fp32-master", action="store_true",
                    help="recipe optimizer numerics: fp32 master + AdamW state, clip 1.0")
    ap.add_argument("--fsdp", action="store_true")
    ap.add_argument("--keep-layers", type=int, default=0,
                    help="recipe LLAVA_KEEP_ACTIVATION_LAYERS (faster_7b.sh sets 32 = all "
                         "layers): the last N decoder layers skip activation checkpointing. "
                         "Same function as full checkpointing (N0), trades memory for the "
                         "recompute.")
    ap.add_argument("--attn", choices=["sdpa", "eager", "splash"], default="sdpa",
                    help="LM attention. The recipe uses flash_attention_2 (train_mem.py), "
                         "which has no XLA kernel; sdpa is the closest same-function path. "
                         "splash routes HF's sdpa interface to Pallas splash (causal, "
                         "no window). Needs the torch_xla pallas extra.")
    args = ap.parse_args()

    mesh = None
    if args.spmd:
        import numpy as np
        import torch_xla.runtime as xr
        import torch_xla.distributed.spmd as xs
        xr.use_spmd()
        n = xr.global_runtime_device_count()
        # Two axis names so Pallas splash (which shards q/k/v over ("data", "fsdp"))
        # can run here; fsdp has size 1, so every annotation below is unchanged.
        mesh = xs.Mesh(np.arange(n), (n, 1), ("data", "fsdp"))

    from transformers import AutoConfig
    from llava.model.language_model.llava_llama import LlavaLlamaForCausalLM
    from llava.constants import IMAGE_TOKEN_INDEX, IGNORE_INDEX
    from tpu_smoke.llava_static_splice import check_supported, static_splice

    cfg = AutoConfig.from_pretrained("lmsys/vicuna-7b-v1.5")
    cfg.num_hidden_layers = args.layers
    cfg.torch_dtype = torch.bfloat16
    if args.attn == "splash":
        from tpu_smoke import hf_splash, splash_config
        if mesh is None:
            raise SystemExit("--attn splash needs --spmd")
        splash_config.set_mesh(mesh, "data")
        hf_splash.register()
        cfg._attn_implementation = "xla_splash"
    else:
        cfg._attn_implementation = args.attn
    torch.manual_seed(args.seed)
    xm.set_rng_state(args.seed)
    model = LlavaLlamaForCausalLM(cfg).to(torch.bfloat16)
    # The recipe's model_args, as train.py hands them to initialize_vision_modules.
    margs = types.SimpleNamespace(
        vision_tower="openai/clip-vit-large-patch14-336", mm_vision_select_layer=-2,
        mm_vision_select_feature="patch", pretrain_mm_mlp_adapter=None,
        mm_patch_merge_type="flat", mm_projector_type="mlp2x_gelu")
    model.get_model().initialize_vision_modules(model_args=margs)
    vt = model.get_vision_tower()
    vt.to(dtype=torch.bfloat16)
    model.config.image_aspect_ratio = "pad"
    model.config.mm_use_im_start_end = False
    vt.requires_grad_(False)                   # recipe: vision tower frozen
    dev = torch_xla.device()
    model.to(dev)
    # Recipe: --gradient_checkpointing True. torch.utils.checkpoint looks up
    # torch.<device_type> for RNG state, so on XLA it raises "module 'torch' has no
    # attribute 'xla'" in both reentrant and non-reentrant mode (11-05).
    # torch_xla ships its own checkpoint with the same signature; HF's
    # _set_gradient_checkpointing takes the function to use, so it is passed in
    # instead of patching transformers.
    import functools
    from torch_xla.utils.checkpoint import checkpoint as xla_checkpoint
    ckpt = functools.partial(xla_checkpoint, use_reentrant=True)
    if args.keep_layers:
        # Same rule as upstream llava_llama.gradient_checkpointing_enable: layers in
        # the retained set run without checkpointing.
        retained = frozenset(model.model.layers[-args.keep_layers:])
        inner = ckpt

        def ckpt(function, *a, **k):
            if getattr(function, "__self__", None) in retained:
                return function(*a, **k)
            return inner(function, *a, **k)
    model._set_gradient_checkpointing(enable=True, gradient_checkpointing_func=ckpt)
    model.config.use_cache = False
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_all = sum(p.numel() for p in model.parameters())

    specs = {}
    if mesh is not None and args.fsdp:
        for prm in model.parameters():
            dims = [d for d in range(prm.dim()) if prm.shape[d] % mesh.size() == 0]
            if dims:
                spec = [None] * prm.dim(); spec[dims[0]] = "data"
                xs.mark_sharding(prm, mesh, tuple(spec))
                specs[prm] = tuple(spec)
    if args.weight_grad_hint:
        import torch.nn.functional as F
        n_hint = 0
        for mod in model.modules():
            w = getattr(mod, "weight", None)
            if w is None or not w.requires_grad or w not in specs:
                continue
            if isinstance(mod, torch.nn.Linear):
                mod.forward = (lambda x, mod=mod, sp=specs[w]:
                               F.linear(x, xs.mark_sharding_with_gradients(mod.weight, mesh, sp), mod.bias))
            elif isinstance(mod, torch.nn.Embedding):
                mod.forward = (lambda x, mod=mod, sp=specs[w]:
                               F.embedding(x, xs.mark_sharding_with_gradients(mod.weight, mesh, sp),
                                           mod.padding_idx, mod.max_norm, mod.norm_type,
                                           mod.scale_grad_by_freq, mod.sparse))
            else:
                continue
            n_hint += 1
        print(f"weight-grad hints on {n_hint} modules", flush=True)
    trainable = [p for p in model.parameters() if p.requires_grad]
    master = None
    if args.fp32_master:
        # Recipe --bf16 True under ZeRO-3 (scripts/train_faster_7b.sh): bf16 compute
        # weights, fp32 master + AdamW state, HF Trainer max_grad_norm 1.0 on the
        # fp32 grads. master[i] MUST pair with trainable[i].
        master = []
        for prm in trainable:
            m = prm.detach().float()
            if prm in specs:
                xs.mark_sharding(m, mesh, specs[prm])
            master.append(m.requires_grad_(True))
    opt = torch.optim.AdamW(master if master is not None else trainable,
                            lr=2e-5 * args.lr_scale, weight_decay=0.0)

    gen = torch.Generator().manual_seed(0)
    ids = torch.randint(1000, 30000, (args.batch, args.seq), generator=gen)
    ids[:, 1] = IMAGE_TOKEN_INDEX
    labels = ids.clone()
    labels[:, :args.seq // 4] = IGNORE_INDEX     # prompt part masked
    labels[:, 1] = IGNORE_INDEX
    images = torch.rand(args.batch, 3, 336, 336, generator=gen).to(torch.bfloat16)
    mask = torch.ones_like(ids, dtype=torch.long)
    image_pos = check_supported(ids, mask)        # host-side, from the data
    from tpu_smoke.llava_static_splice import uniform_image_pos
    shared_pos = uniform_image_pos(image_pos) if args.splice_concat else None
    if args.splice_concat and shared_pos is None:
        raise ValueError("--splice-concat: <image> positions differ across the batch")
    ids, labels, images = ids.to(dev), labels.to(dev), images.to(dev)
    mask, image_pos = mask.to(dev), image_pos.to(dev)
    if mesh is not None:
        for t in (ids, labels, mask):
            xs.mark_sharding(t, mesh, ("data", None))
        xs.mark_sharding(images, mesh, ("data", None, None, None))

    def one_step():
        opt.zero_grad(set_to_none=True)
        # Upstream's per-sample splice (llava_arch.py:145) replicated the global
        # batch on every chip at 64 chips, and with its outputs re-annotated it
        # aborted the SPMD partitioner (11-06). static_splice computes the same
        # tensors with static batched ops; bitwise-equal to upstream on CPU, see
        # tpu_smoke/test_llava_static_splice.py (mutations caught).
        pos, am, emb, lab = static_splice(model, ids, mask, labels, images, image_pos,
                                          uniform_pos=shared_pos)
        if mesh is not None:
            xs.mark_sharding(emb, mesh, ("data", None, None))
            xs.mark_sharding(lab, mesh, ("data", None))
            xs.mark_sharding(am, mesh, ("data", None))
            xs.mark_sharding(pos, mesh, ("data", None))
        out = model(input_ids=None, attention_mask=am, position_ids=pos,
                    inputs_embeds=emb, labels=lab)
        out.loss.backward()
        if master is not None:
            for m, prm in zip(master, trainable):
                if args.shard_grads and prm.grad is not None and prm in specs:
                    xs.mark_sharding(prm.grad, mesh, specs[prm])
                m.grad = prm.grad.float() if prm.grad is not None else None
            torch.nn.utils.clip_grad_norm_(master, 1.0)
            opt.step()
            with torch.no_grad():
                for m, prm in zip(master, trainable):
                    prm.copy_(m)
                    prm.grad = None
        else:
            opt.step()
        return out.loss.detach()

    if args.traj:
        import numpy as np
        succ = np.random.default_rng(1234).integers(1000, 30000, size=(30000, 4))

        def text_batch(g):
            cur = g.integers(1000, 30000, size=args.batch)
            out = np.empty((args.batch, args.seq), dtype=np.int64)
            for t in range(args.seq):
                out[:, t] = cur
                cur = succ[cur, g.integers(0, 4, size=args.batch)]
            return torch.from_numpy(out)
        traj = []
        for step in range(args.traj):
            g = np.random.default_rng([args.seed, step])
            ids = text_batch(g)
            ids[:, 1] = IMAGE_TOKEN_INDEX
            labels = ids.clone()
            labels[:, :args.seq // 4] = IGNORE_INDEX
            labels[:, 1] = IGNORE_INDEX
            images = torch.rand(args.batch, 3, 336, 336,
                                generator=torch.Generator().manual_seed(args.seed * 1000003 + step)).to(torch.bfloat16)
            mask = torch.ones_like(ids, dtype=torch.long)
            image_pos = check_supported(ids, mask)
            ids, labels, images = ids.to(dev), labels.to(dev), images.to(dev)
            mask, image_pos = mask.to(dev), image_pos.to(dev)
            if mesh is not None:
                for t in (ids, labels, mask):
                    xs.mark_sharding(t, mesh, ("data", None))
                xs.mark_sharding(images, mesh, ("data", None, None, None))
            loss = one_step(); torch_xla.sync()
            traj.append([float(loss), None])
        print("TRAJ " + json.dumps(dict(seed=args.seed, lr_scale=args.lr_scale, attn=args.attn,
                                        splice_concat=args.splice_concat, traj=traj)), flush=True)
        return

    losses = []
    t0 = time.time()
    for _ in range(args.warmup):
        loss = one_step(); torch_xla.sync(); xm.wait_device_ops(); losses.append(float(loss))
    warm_s = time.time() - t0
    windows = []
    for _ in range(args.windows):
        u0 = uncached(); tr, sb, dv = [], [], []
        for _ in range(args.window):
            a = time.perf_counter(); loss = one_step(); b = time.perf_counter()
            torch_xla.sync(); c = time.perf_counter(); xm.wait_device_ops(); d = time.perf_counter()
            tr.append(1000 * (b - a)); sb.append(1000 * (c - b)); dv.append(1000 * (d - c))
        last = float(loss); u1 = uncached()
        windows.append(dict(trace_ms=round(statistics.median(tr), 2),
                            submit_ms=round(statistics.median(sb), 2),
                            device_ms=round(statistics.median(dv), 2),
                            total_ms=round(statistics.median([a + b + c for a, b, c in zip(tr, sb, dv)]), 2),
                            tainted=bool(u1 - u0) if None not in (u0, u1) else None))
    # Pipelined window: what a real training loop pays. Every window above
    # blocks per step (TRACE+SUBMIT+DEVICE, each step from an idle device);
    # here the host only cuts the graph with sync() and traces step k+1 while the
    # device runs step k, so the step costs about max(host, device).
    # The result is kept live and the device drained once at the end.
    from tpu_smoke.step_counters import Probe
    probe = Probe(); probe.start()
    t_a = time.perf_counter()
    for _ in range(args.window):
        res = one_step()
        torch_xla.sync()
    xm.wait_device_ops()
    pipelined_ms = round(1000 * (time.perf_counter() - t_a) / args.window, 2)
    probe_out = probe.report(args.window)
    if args.profile:
        # Device attribution (11-06): xplane trace of 3 steady-state steps, read
        # offline with xprof (tools: hlo_op_profile / framework_op_stats).
        import torch_xla.debug.profiler as xp
        os.makedirs(args.profile, exist_ok=True)
        xp.start_trace(args.profile)
        for _ in range(3):
            res = one_step()
            torch_xla.sync()
        xm.wait_device_ops()
        xp.stop_trace()
    _ = [float(r) for r in (res if isinstance(res, tuple) else (res,))]
    clean = [w for w in windows if not w["tainted"]] or windows
    tot = [w["total_ms"] for w in clean]
    print("SUMMARY " + json.dumps(dict(
        project="LLaVA", shard_grads=args.shard_grads, weight_grad_hint=args.weight_grad_hint, splice_concat=args.splice_concat, fp32_master=args.fp32_master, attn=args.attn, keep_layers=args.keep_layers, layers=args.layers, batch=args.batch, seq=args.seq,
        chips=int(mesh.size()) if mesh is not None else 1, fsdp=args.fsdp,
        params=n_all, trainable=n_train, warmup_s=round(warm_s, 1),
        warmup_losses=[round(v, 4) for v in losses],
        tainted=[w["tainted"] for w in windows], pipelined_ms=pipelined_ms, **probe_out,
        trace_ms=statistics.median(w["trace_ms"] for w in clean),
        submit_ms=statistics.median(w["submit_ms"] for w in clean),
        device_ms=statistics.median(w["device_ms"] for w in clean),
        total_ms=statistics.median(tot),
        spread=round((max(tot) - min(tot)) / statistics.median(tot), 3),
        last_loss=repr(last))), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print("FAILED " + type(exc).__name__ + ": " + str(exc)[:3000], flush=True)
        raise
