# tpu_smoke: LLaVA-1.5-7B training step on TPU (torch_xla)

Scripts behind [TPU v6e results](../docs/tpu-v6e-results.md). They import the `llava` package of this repository unchanged and replace only what does not run on an XLA SPMD mesh, from outside the upstream files. The language model is randomly initialised and the batches are synthetic, so these scripts measure code paths, step time and short loss trajectories, not training quality.

## Files

| File | What it is |
|---|---|
| `llava_step.py` | Entry point. Builds `LlavaLlamaForCausalLM` with the 7B recipe wiring, shards it over all chips, runs warmup, blocking windows and one pipelined window, and prints a `SUMMARY` JSON line. `--traj N` prints a `TRAJ` loss trajectory for the statistical gate instead. |
| `llava_static_splice.py` | Batched, static-shape replacement for upstream `prepare_inputs_labels_for_multimodal` (one `<image>` per sample, right padding). The general path uses gather and `where`; `--splice-concat` uses slice and concatenation when every row has its `<image>` at the same position. Unsupported batches raise. |
| `hf_splash.py` | Registers torch_xla's Pallas splash attention as a HuggingFace attention key (`xla_splash`), padding the sequence at the end to a multiple of `SPLASH_PAD_TO`. |
| `splash_config.py` | Splash mesh state, `SplashAttentionConfig` builder and the Mosaic IR version pin required by libtpu 0.0.21. |
| `step_counters.py` | Per-step torch_xla counter deltas (executions, host transfers, CPU fallbacks, compiles) and Python GC time for the pipelined window. |
| `gate_eval.py` | Statistical gate over paired-seed `TRAJ` lines: PASS / INCORRECT / UNRESOLVED. Pure Python. |
| `test_llava_static_splice.py` | CPU, bitwise: `static_splice` (both paths) equals upstream on position ids, attention mask, input embeddings and labels; wrong-position mutant and unsupported inputs are rejected. |
| `test_gate_eval.py` | CPU: `gate_eval.py` returns PASS, INCORRECT and UNRESOLVED on synthetic trajectories built for each case. |
| `test_hf_splash_equiv.py` | TPU, one host with 4 chips: splash forward and q/k/v gradients match HF sdpa within bf16 tolerance at T=2623; a padded mask is refused. |

## Environment

TPU runs used torch 2.9.0, torch_xla 2.9.0 with its Pallas extra (jax 0.7.1), libtpu 0.0.21 and transformers 4.56.2. Upstream pins torch 2.1.2 and transformers 4.37.2, which cannot share an environment with torch_xla 2.9, so use a separate environment. The `llava` package imports under transformers 4.56.2 without changes. DeepSpeed, flash-attn, Apex and Liger are not imported by these scripts.

`llava_step.py` loads the `lmsys/vicuna-7b-v1.5` config and the `openai/clip-vit-large-patch14-336` weights from the HuggingFace Hub. Point `HF_HOME` at a cache that has them or can download them.

## CPU tests

From the repository root. Neither test imports torch_xla; checked with torch 2.5.1 and transformers 4.49.0.

```bash
python -m tpu_smoke.test_llava_static_splice   # needs torch + transformers
python -m tpu_smoke.test_gate_eval             # pure Python
```

On one TPU host with 4 chips:

```bash
PJRT_DEVICE=TPU python -m tpu_smoke.test_hf_splash_equiv
```

## Running on a 64-chip slice

Run the same command on every host of the slice from the repository root, for example with `gcloud compute tpus tpu-vm ssh <tpu-name> --zone <zone> --worker=all --command '...'`. Each arm of an A/B/A comparison is a separate launch, in the order baseline, candidate, baseline. The environment used for the reported runs:

```bash
ulimit -c 0                       # a core dump of a crashed 31 GB process held the TPU devices for 30+ minutes
export PJRT_DEVICE=TPU
export TORCHDYNAMO_DISABLE=1
export ALLOW_MULTIPLE_LIBTPU_LOAD=1
export HF_HOME=<hf-cache>
export SPLASH_PAD_TO=512          # read by --attn splash: 2,623 tokens are padded to 3,072
cd <repo-root>
python -m tpu_smoke.llava_step <flags> > <log-dir>/<arm>.log 2>&1
```

In SPMD every step waits for the slowest host, so CPU load from other jobs on any host raises the step time. Check the load on all hosts before measuring.

### Timing configurations (2,623 tokens per sample)

`--seq` counts text tokens including the `<image>` token: 2,047 text tokens plus 576 image tokens give 2,623.

```bash
COMMON="--spmd --fsdp --fp32-master --layers 32 --batch 128 --seq 2048 --warmup 4 --window 8 --windows 3"

python -m tpu_smoke.llava_step $COMMON --attn sdpa                          # baseline
python -m tpu_smoke.llava_step $COMMON --attn splash                        # + splash attention
python -m tpu_smoke.llava_step $COMMON --attn splash --splice-concat        # adopted configuration
```

The baseline and the splash-only arm use the general gather path of `static_splice`. Upstream's per-sample splice does not run on the 64-chip mesh (see the results document), so no arm uses it.

Bucket lengths for the real-length estimate (adopted configuration): `--seq 961` gives 1,536 tokens, `--seq 1473` gives 2,048 tokens.

Rejected gradient-communication attempts, each an A/B/A against the adopted configuration at `--seq 961`:

```bash
python -m tpu_smoke.llava_step $COMMON_1536 --shard-grads
python -m tpu_smoke.llava_step $COMMON_1536 --weight-grad-hint
LIBTPU_INIT_ARGS="--xla_tpu_enable_all_reduce_scatter_fusion=true --xla_tpu_enable_data_parallel_all_reduce_opt=true --xla_tpu_enable_async_collective_fusion_fuse_all_reduce=true" \
  python -m tpu_smoke.llava_step $COMMON_1536
```

where `COMMON_1536="--spmd --fsdp --fp32-master --layers 32 --batch 128 --seq 961 --warmup 4 --window 8 --windows 3 --attn splash --splice-concat"`.

### Reading the output

The last line is `SUMMARY {json}` or `FAILED <error>`. Fields:

- `pipelined_ms`: mean step time of the pipelined window (the host traces step k+1 while the device runs step k). This is the reported step time.
- `trace_ms`, `submit_ms`, `device_ms`, `total_ms`: medians over the blocking windows, where every step starts from an idle device.
- `spread`: (max − min) / median of the blocking-window totals.
- `tainted`: per blocking window, whether a compilation happened inside it; tainted windows are excluded when a clean one exists.
- `counters_per_step`, `gc_ms_per_step`: from `step_counters.py`, over the pipelined window.
- `warmup_losses`, `last_loss`.

### Statistical gate

Three arms × seeds 0, 1, 2, 120 steps each, at the 1,536-token bucket:

```bash
GATE="--spmd --fsdp --fp32-master --layers 32 --batch 128 --seq 961 --traj 120"
for seed in 0 1 2; do
  python -m tpu_smoke.llava_step $GATE --seed $seed --attn sdpa                      > <log-dir>/up_s$seed.log   2>&1
  python -m tpu_smoke.llava_step $GATE --seed $seed --attn splash --splice-concat    > <log-dir>/cand_s$seed.log 2>&1
  python -m tpu_smoke.llava_step $GATE --seed $seed --attn sdpa --lr-scale 1.02      > <log-dir>/sens_s$seed.log 2>&1
done
```

Then, on one host, collect one line per arm and seed in the format `gate_eval.py` parses (`<tag> <arm> s<seed> TRAJ {...}`) and evaluate:

```bash
for seed in 0 1 2; do for arm in up cand sens; do
  echo "w0 $arm s$seed $(grep -m1 '^TRAJ' <log-dir>/${arm}_s$seed.log)"
done; done > gate_llava.txt
python -m tpu_smoke.gate_eval gate_llava.txt        # defaults: --base up --skip 10 --windows 5
```

The gate counts only if the sensitivity arm `sens` (learning rate × 1.02) is judged INCORRECT.
