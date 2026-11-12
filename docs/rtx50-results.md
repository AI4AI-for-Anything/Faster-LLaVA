# RTX 50 series results

LLaVA-1.5-7B stage-two full fine-tuning on **8×RTX 5080 (16 GB)** with the [RTX 50 recipe](../scripts/train_rtx50_7b.sh): one process drives all GPUs through [RoundPipe](https://github.com/thustorage/RoundPipe), with parameters, gradients and AdamW state in host memory. This page records the timing protocol, the numerical checks and what is still pending. These results do **not** establish lossless acceleration.

## Time per optimizer step

| Implementation on 8×RTX 5080 | Time / optimizer step (ms) ↓ | Speedup ↑ | Peak GPU memory |
|---|---:|---:|---:|
| Upstream LLaVA (ZeRO-3 + CPU AdamW offload, 4 × accumulation 4) | 51,231 | 1.00× | 13.1 GiB |
| Faster-LLaVA H20 recipe (ZeRO-3 + CPU AdamW offload, 8 × accumulation 2) | 28,277 | 1.81× | 10.4 GiB |
| **RTX 50 recipe (RoundPipe)** | **11,475** | **4.46×** | **5.8 GiB** |

Each row is the mean of independent runs of `bench.py time` on the same fixture and machine: upstream 51,409 / 51,049 / 51,236 ms, Faster-LLaVA 28,726 / 27,828 ms, RTX 50 recipe 11,519 / 11,430 ms. A fourth upstream run (52,820 ms) is left out of the mean because its second window was 9% slower than its first (five consecutive slow steps of unknown cause). All runs keep global batch 128 and the step membership of the reference sampler.

For comparison only, on other hardware: Faster-LLaVA on **4×H20** takes 8,777 ms per step against a 20,934 ms baseline ([README](../README.md)).

### Protocol

- **Workload.** The fixed timing fixture of [ResearchBoost-Agent's LLaVA task](https://github.com/DAGroup-PKU/ResearchBoost-Agent/tree/main/tasks/llava): 18 optimizer steps × 128 samples drawn from `llava_v1_5_mix665k.json` with the RBA sampling (17 image-text steps and 1 text-only step). Vicuna-7B v1.5, CLIP ViT-L/14-336 (frozen), the official stage-one projector, `model_max_length` 2048, learning rate 2e-5, cosine schedule with warmup ratio 0.03, weight decay 0, bf16 + TF32, fp32 master weights and AdamW state, no gradient clipping (as `scripts/zero3.json`).
- **Timing.** Steps 1–2 warm up; the time per step is the mean of two 8-step windows (steps 3–10 and 11–18). Each run is a fresh process; GPU jobs are serialised by a host lock and no other job runs during a timing run. Runs whose two windows differ by more than 10% are not used.
- **Upstream arm.** haotian-liu/LLaVA at `c121f04`, `train_mem.py` with FlashAttention-2, DeepSpeed ZeRO-3 from `scripts/zero3.json` plus CPU AdamW offload (16 GB cards cannot hold the ZeRO-3 shards and Adam state otherwise). Micro-batch 4 × accumulation 4 per GPU; micro-batch 8 runs out of memory.
- **Faster-LLaVA arm.** This repository's `main` with the switches of `scripts/configs/faster_7b.sh` and the same DeepSpeed configuration, micro-batch 8 × accumulation 2. `LLAVA_HEAD_LIGER_CE` is off: with liger-kernel 0.8.5, `fused_head_loss.py` passes arguments in the older positional order and fails.
- **Environment.** Python 3.12, PyTorch 2.14.1+cu132, Transformers 4.37.2, FlashAttention 2.8.3 built from source for sm_120, DeepSpeed 0.19.9 (baseline arms only), RoundPipe `fix/v0.1.2` (see below). Host: 2× AMD EPYC 7K62, 503 GB RAM, PCIe 4.0 without GPU peer-to-peer.

## What the recipe changes

| Change | Where | Effect on the fixture |
|---|---|---|
| Fixed RoundPipe stage plan (`--stage_plan 3,1,2`): 3 decoder layers per forward stage, CLIP + projector + 2 layers in the first, one layer per backward stage | `train_rtx50.py` | RoundPipe re-plans from measured layer times during the first steps; each new layout allocates new pinned buffers and the first step on it took 20–40 s longer. The light first stage stops CLIP from throttling the forward pipeline (−3%). |
| 17 token-balanced microbatches (2 × GPUs + 1) | `train_rtx50.py` | Shorter pipeline fill and drain (−8% against 9). 18 and 25 were slower. |
| `F.rms_norm` (`LLAVA_RTX50_FUSED_RMSNORM=1`) | `rtx50_sequential.py` | One fused kernel each way instead of about a dozen per norm (−5%). |
| Layer metadata as one int32 tensor | `rtx50_sequential.py` | RoundPipe moves the inter-layer state through host memory at every stage boundary; the metadata now travels as one tensor per microbatch instead of four (≈ 6.1k of the 10.9k copies per step were such small tensors). Bitwise identical; its own effect was not timed separately. |

RoundPipe launches the kernels of all eight GPUs from Python threads of one process, so host launch cost bounds the step: per-layer launches matter more than per-kernel GPU time.

### Tried and rejected

| Attempt | Result |
|---|---|
| Liger RMSNorm / RoPE / SwiGLU kernels | 12.5% less kernel time but a slower step (12.45 vs 12.21 s): each Triton launch costs 66–85 µs of host time, and with eight launching threads the GPUs starve (22 GPU-s per step host-starved vs 5). |
| RoundPipe's automatic stage planner | 40 s re-plan steps; replaced by the fixed plan. |
| 18 or 25 microbatches; 2- or 4-layer forward stages | No gain or slower. |
| Rotary rows gathered once per microbatch | Bitwise identical, but no resolvable gain in paired timing (11,461 vs 11,475 ms); removed. |
| Overlapping the CPU AdamW step with backward | Would need a RoundPipe feature (per-layer optimizer steps); not attempted. The serial CPU optimizer phase is ≈ 2.2 s of the 11.5 s step. |

### RoundPipe fixes

The recipe depends on two bug fixes on RoundPipe's `fix/v0.1.2` branch:

- the optimizer thread sized its OpenMP team from `torch.get_num_threads()` of a new thread, which is 1 once a `DataLoader(pin_memory=True)` has started; the CPU AdamW then ran on one thread (11–22 s per step instead of ≈ 4.5 s);
- `import roundpipe` failed on hosts without a visible GPU.

## Numerical checks

- **Parity.** A two-layer random LLaVA through the packed RoundPipe path matches upstream `LlavaLlamaForCausalLM` (loss within 1.4e-5 relative, worst gradient cosine 0.99963) with and without fused RMSNorm.
- **Statistical gate (ResearchBoost-Agent rule).** Six seeded 60-step trajectories of the upstream arm fix δ = 2σ for two framework-neutral endpoints, measured by the same upstream evaluator on saved weights: held-out probe loss (δ = 0.0085) and a 128-sample image-text loss (δ = 0.0235). Three paired seeds of the RTX 50 recipe: **PASS**. The image-text mean difference is −0.0010 (−0.09σ). The probe mean difference is +0.0054 (+1.28σ, p = 0.04): inside δ, but positive in all three pairs.
- **Known differences from upstream.** Samples are packed without padding and the loss is the mean over all supervised tokens of the optimizer step (as `LLAVA_GLOBAL_TOKEN_LOSS=1` on the H20 path); RoundPipe accumulates gradients across microbatches in bf16; `F.rms_norm` rounds once instead of twice. These change the numerical trajectory.

## Full-training results

Pending. One epoch on all 665,298 records (5,198 optimizer steps) with `scripts/train_rtx50_7b.sh` is running on 8×RTX 5080. GQA testdev, TextVQA val, POPE, ScienceQA-IMG and the SEED image questions will be evaluated with the upstream `scripts/v1_5/eval` protocol, next to the official LLaVA-v1.5-7B checkpoint rerun under the same protocol. The SEED video questions (5,009 of the historical 19,242) are not scored: the current SEED-Bench release ships different video frames. VizWiz, MMBench and MM-Vet are not scored locally.

## Source and reproduction

See the [README for RTX 50 series GPUs](../README_5080.md), the [training launcher](../scripts/train_rtx50_7b.sh), its [configuration](../scripts/configs/rtx50_7b.sh) and the [changed files](change-views/rtx50-files.md).
