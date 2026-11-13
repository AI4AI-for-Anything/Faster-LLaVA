# TPU v6e training-step results

Training-step timing of LLaVA-1.5-7B on Google Cloud TPU v6e with PyTorch/XLA. This is an experimental port: the language model is randomly initialised and all inputs are synthetic.

On **64 TPU v6e chips**, at **global batch 128** with fp32 master weights and gradient clipping, two changes take the steady-state training step from **1931 ms to 1134 ms (1.70×)** on synthetic 2,623-token samples. Both changes are numerical tier N1, and the combination passed a 120-step paired-seed statistical gate. These results do **not** establish training quality or lossless acceleration, and make **no** speed comparison with the H20 results.

| Configuration | Time / step (ms) ↓ | Speedup ↑ |
|---|---:|---:|
| Baseline: HF SDPA attention, gather splice | 1931.03 | 1.00× |
| + Pallas splash attention | 1356.31 | 1.42× |
| **+ static image splice (adopted)** | **1134.31** | **1.70×** |

Each step is measured baseline–candidate–baseline in independent processes. The two optimizations were measured in two separate A/B/A batches. The shared configuration (splash, gather splice) measured 1356.31 ms in the first batch and 1355.77 ms in the second, and 1.70× is the first batch's baseline over the second batch's candidate.

## Hardware and software

| Setting | Value |
|---|---|
| Accelerator | Google Cloud TPU v6e, one slice of 64 chips on 16 hosts |
| Software | torch 2.9.0, torch_xla 2.9.0, libtpu 0.0.21, transformers 4.56.2 |
| Attention kernel | torch_xla Pallas splash attention (Pallas extra with jax 0.7.1) |
| Model code | the `llava` package of this repository, imported unchanged |

Upstream pins torch 2.1.2 and transformers 4.37.2, which cannot be installed together with torch_xla 2.9. The `llava` package imports under transformers 4.56.2 without changes.

## Workload

| Setting | Value |
|---|---|
| Model | `LlavaLlamaForCausalLM` with the 7B recipe wiring: Vicuna-7B config (32 layers), mlp2x_gelu projector, vision select layer −2, image aspect ratio `pad` |
| Weights | language model and projector randomly initialised; pretrained CLIP ViT-L/14-336, frozen |
| Parameters | 7.06 B total, 6.76 B trainable |
| Global batch | 128 samples per optimizer step, no gradient accumulation |
| Layout | one 64-way mesh axis: the batch is split 2 samples per chip; parameters and fp32 master weights are sharded over the same axis (FSDP-style) |
| Sequence | 2,047 text tokens + 576 image tokens = 2,623 per sample, fixed length |
| Precision and optimizer | bf16 compute weights; fp32 master weights and AdamW state; gradient-norm clip 1.0, as in the ZeRO-3 + `--bf16 True` recipe |
| Activation checkpointing | every decoder layer, as in the recipe (`--gradient_checkpointing True`), through torch_xla's checkpoint |
| Data | synthetic: random token ids with one `<image>` token at position 1, labels masked on the first quarter of the text, random 336×336 images; the same batch every step |

**Batch fairness.** Baseline and candidate use the same per-chip batch and the same 64-chip layout (global 128 / 64 chips = 2 samples per chip), so every ratio in this document is like-for-like on TPU. These numbers are **not** comparable with the GPU numbers in [H20 results](h20-results.md): the GPU recipe runs 16 samples per GPU on 8 GPUs (the 2.39× benchmark used 16 per GPU on 4 GPUs with accumulation 2), so the per-device batch differs. No TPU-versus-GPU speed claim is made.

## Timing method

- Each arm is an independent process launch, in the order baseline (A), candidate (B), baseline (A2).
- In each process: 4 warmup steps (including compilation); three blocking windows of 8 steps, in which every step starts from an idle device (trace, submit and device time are taken per step, and each window reports medians); then one pipelined window of 8 steps, in which the host traces step k+1 while the device runs step k. **The reported step time is the mean of the pipelined window.**
- Window range is (max − min) / median of the three blocking-window totals. An arm counts only with range ≤ 10%; windows that compiled are excluded.
- A candidate passes when its speedup over the baseline is at least max(1.02, 1 + 2 × the difference between the two baselines).

## Results per optimization

### Pallas splash attention

`hf_splash.py` registers splash attention as a HuggingFace attention key, so the upstream model uses it without edits. The sequence is padded at the end from 2,623 to 3,072, so 512-wide blocks fit. Under the causal mask the padded keys are never attended, and the padded query rows are dropped before returning.

| Arm | Pipelined ms ↓ | Blocking total | Trace | Device | Window range | Final loss |
|---|---:|---:|---:|---:|---:|---|
| A HF SDPA | 1931.03 | 1950.1 | 287.0 | 1621.7 | 0.1% | 6.157079696655273 |
| **B splash** | **1356.31** | 1652.4 | 302.2 | 1299.6 | 0.2% | 6.141944885253906 |
| A2 HF SDPA | 1933.77 | 1950.2 | 287.9 | 1609.5 | 0.1% | 6.157079696655273 |

Speedup **1.42×**, baseline difference 0.14%: passes.

Part of this gain is a host synchronization rather than the kernel. transformers' sdpa mask path calls `padding_mask.all()` on a 2-D mask, which fetches a value from the device every step; the custom splash key does not take that path. In a diagnostic on 4 chips with a 4-layer model, the SDPA arm ran 2 executions per step with one `aten::_local_scalar_dense`, and the splash arm ran 1. Step time fell from 286.3 to 223.6 ms, while device time fell by only 22 ms. The A/B/A therefore compares the HF SDPA path with the splash path as a whole.

Block size matters. In an earlier diagnostic configuration (global batch 64, no fp32 master, no clipping), padding only to a multiple of 128 (2,688 tokens) forced 128-wide blocks and made splash 8.6% slower than SDPA.

### Static image splice

Upstream `prepare_inputs_labels_for_multimodal` splices image features into each sample in a Python loop and re-stacks the batch. On the 64-chip mesh its output lost the batch sharding, so every chip computed the global batch: in an early run at global batch 64 that needed a 57.8 GB attention-score buffer against 33.8 GB per chip. Re-annotating its outputs instead crashed the XLA SPMD partitioner (`IsScalarWithElementType` check). All arms in this document therefore use `llava_static_splice.static_splice`, a batched re-implementation that is bitwise equal to upstream on CPU for the LLaVA-1.5 case (one `<image>` per sample, right padding). Other batches raise.

A device profile of the splash configuration (xplane trace of 3 steady-state steps read with xprof, global batch 128, fp32 master, traced on one host's 4 chips) attributed the device time as follows:

| Item | Share of device self time |
|---|---:|
| splash attention (forward, backward, checkpoint recompute) | ~25% |
| gather fusions from the splice | **~17% (about 220 ms per step)** |
| all-reduce / all-gather | 1–2% |
| other (matmul fusions and the rest) | ~55% |

On TPU, XLA expands the splice's per-row `torch.gather` into a dense compare–select–reduce over all position pairs, in the forward and the backward. The LLaVA-1.5 preprocessing moves `<image>` to the start of its sentence (`llava/train/train.py:318-321`), so within a batch every sample has its image at the same position. With one shared position the splice becomes static slicing and concatenation (`--splice-concat`). The shared position is checked on the host, and a batch that violates it raises.

| Arm | Pipelined ms ↓ | Trace | Device | Window range | Step-0 loss | Final loss |
|---|---:|---:|---:|---:|---|---|
| A gather splice | 1355.77 | 302.1 | 1307.5 | 0.2% | 11.1888 | 6.141944885253906 |
| **B static splice** | **1134.31** | 309.0 | 1075.9 | 0.2% | 11.1890 | 6.128485679626465 |
| A2 gather splice | 1355.89 | 302.6 | 1306.7 | 0.1% | | 6.141944885253906 |

Speedup **1.195×**, baseline difference 0.01%: passes. Device time fell by 231 ms, consistent with the ~220 ms the profile attributed to the gather fusions.

### Diagnostic only: no activation recomputation

Skipping activation checkpointing in all 32 layers (recipe `LLAVA_KEEP_ACTIVATION_LAYERS=32`) was measured only in the earlier diagnostic configuration (global batch 64, no fp32 master, no clipping): 1059.3 / 1060.4 / 1069.2 ms for A / B / A2, with bitwise-identical losses. No benefit; not re-measured under the conditions above.

## Numerical tier

N0 means bitwise identical to the baseline. N1 means the same function in the same dtypes, but with a different accumulation order or fusion, so not bitwise identical.

- **Splash attention: N1** (online softmax). On 4 chips at H=32, D=128, T=2,623, bf16, with HF's 4-D causal mask, the maximum relative error against HF sdpa is 3.95e-3 (output), 4.42e-3 (dq), 4.96e-3 (dk) and 5.35e-3 (dv) (`tpu_smoke/test_hf_splash_equiv.py`). The padded-mask refusal in `hf_splash.py` is exercised only by this test: in the model path HF passes no mask to a custom attention key. Right-padded batches remain correct because causal attention never reaches trailing padding.
- **Static splice: N0 on CPU, N1 on TPU.** Both splice paths equal upstream bitwise on CPU (`tpu_smoke/test_llava_static_splice.py`, including padded rows and mutants that must fail). On TPU the step-0 loss already differs (11.1888 vs 11.1890), because the whole-step graph changes and XLA's layout and fusion change with it.

The combination is N1, so it went through the statistical gate.

## Statistical gate

| Setting | Value |
|---|---|
| Hardware and precision | 64 chips, global batch 128, fp32 master weights |
| Sequence | 1,536-token bucket (`--seq 961`: 960 text + 576 image tokens) |
| Data | a fresh synthetic batch every step, drawn from (seed, step); text follows one fixed sparse Markov chain (4 successors per token, shared by all seeds), so the model has something to learn |
| Seeds | 0–2, paired: the seed also sets model initialization and XLA RNG, and each candidate run is compared with the baseline run of the same seed |
| Arms | baseline (HF SDPA + gather splice), candidate (splash + static splice), sensitivity control (baseline with learning rate × 1.02) |

Endpoints are the mean training loss over 5 consecutive windows after the first 10 steps. For each endpoint, δ = 2 × the baseline's standard deviation across seeds, fixed before the candidates are evaluated. An arm is PASS when the mean paired difference is within δ at every endpoint, and INCORRECT when an endpoint exceeds δ with a paired t-test p < 0.05; otherwise it is UNRESOLVED. The gate is valid only if the sensitivity control is judged INCORRECT.

**60 steps: gate invalid.** Loss fell from 11.19 to about 5.8.

| Arm | Verdict | Paired difference per endpoint (δ = 0.048, 0.139, 0.256, 0.209, 0.105) |
|---|---|---|
| Candidate: splash + static splice | PASS | +0.0002, −0.0020, −0.0260, −0.0482, −0.0395 (p=0.21) |
| lr × 1.02 | PASS (should be INCORRECT) | +0.0002, +0.0008, −0.0328, −0.0770 (p=0.09), −0.0959 (p=0.016) |

The learning-rate effect grew with the step count and was significant at the last endpoint, but stayed within δ. The sensitivity control was not detected, so this gate cannot be used and the candidate's PASS here carries no weight.

**120 steps: gate valid, candidate PASS.** Baseline loss fell from 9.75 to 2.05.

| Arm | Verdict | Paired difference per endpoint (δ = 0.104, 0.216, 0.035, 0.042, 0.024) |
|---|---|---|
| **Candidate: splash + static splice** | **PASS** | −0.0026, −0.0403, −0.0210, −0.0013, +0.0015 |
| lr × 1.02 | INCORRECT (sensitivity control detected) | −0.0010, −0.0655, −0.0942 (p=0.003), −0.0651 (p=0.002), −0.0253 (p=0.010) |

## Real-length bucketing: an estimate

The 2,623-token samples above are longer than real data. Lengths of the public `llava_v1_5_mix665k.json` were computed with the v1 conversation template and the Vicuna tokenizer, counting `<image>` as 576 patches and truncating at 2,048:

| Records | Count | p50 | p90 | p95 | p99 | Mean |
|---|---:|---:|---:|---:|---:|---:|
| With image | 624,610 | 759 | 1121 | 1140 | 1327 | 839 |
| Text only | 40,688 | 724 | 1627 | 1805 | 1970 | 830 |

The recipe draws 128 same-modality samples per step (`get_modality_length_grouped_indices`), and an SPMD step has a single shape, so a step pads to its longest sample, rounded up to a bucket. Splash pads to a multiple of 512, so the buckets are {1024, 1536, 2048}. Simulating the recipe's sampling on these lengths put 67.1% of image steps in the 1,536 bucket, 32.9% in 2,048 and none in 1,024. Text-only steps land almost entirely in 2,048. The adopted configuration at the two bucket lengths:

| Total length (incl. 576 image tokens) | Pipelined ms | Trace | Device | Window range |
|---|---:|---:|---:|---:|
| 1,536 | 763.76 | 295.9 | 719.9 | 0.1% |
| 2,048 | 908.41 | 294.5 | 865.3 | 0.3% |

Weighted by the simulated distribution: 0.671 × 763.8 + 0.329 × 908.4 ≈ **811 ms/step**, 28% below the 1134 ms at 2,623 tokens. This is a weighted average of two fixed-length synthetic runs, not a measurement on real data. The SDPA baseline was not measured at these lengths, so no speedup is claimed for real lengths. A 25% shorter sequence cut the step by only 16%; extrapolating linearly leaves about 330 ms/step that does not scale with length. Its source has not been profiled.

## Remaining bottleneck

The adopted configuration is device-bound: at 2,623 tokens, trace takes 309.0 ms and device 1075.9 ms. In a device profile of the adopted configuration at the 1,536-token bucket, about **21% of device time is a full all-reduce of the weight gradients**. With FSDP sharding, a reduce-scatter should be enough, but the SPMD partitioner all-reduces the full gradient and then slices it. This 21% has lower confidence than the timing tables: its per-op breakdown is not reproduced here, and the earlier profile above, taken before the splice change and at 2,623 tokens, put all collectives at 1–2%.

Three attempts to turn the all-reduce into a reduce-scatter, each an A/B/A against the adopted configuration at the 1,536-token bucket, all failed:

| Attempt | A ms | B ms | A2 ms | Result |
|---|---:|---:|---:|---|
| `--shard-grads`: annotate each gradient with its parameter's sharding after backward | 763.45 | 764.90 | 763.49 | No effect; XLA ignores the annotation. Loss bitwise identical (4.597087383270264) |
| libtpu collective flags in `LIBTPU_INIT_ARGS` (see `tpu_smoke/README.md`) | 763.61 | 763.52 | 763.63 | No effect; loss bitwise identical |
| `--weight-grad-hint`: `mark_sharding_with_gradients` on the weights of 228 trainable Linear/Embedding modules | 767.9 | 1080.0 | 765.2 | Device time not lower (763.8 vs 738.5 / 732.2); host trace rose from 296.5 / 296.9 to 790.1 ms. Arm B window range 19.9%, invalid. Loss bitwise identical |

Two directions remain, both outside this port and both estimates: a hand-written gradient collective in the optimizer data flow (manual sharding), worth about 10% of the step; and sequence packing of real data, worth about 2×. Packing breaks the shared image position and would bring back the dynamic splice.

## Limitations

- **Synthetic data, random language model.** Timing runs repeat one fixed batch, and gate runs use Markov-chain text with random images. Nothing here measures training quality.
- **No long training and no downstream evaluation on TPU.** The only quality evidence is the 120-step, 3-seed gate at the 1,536-token bucket. It detects a 2% learning-rate change, and it does not cover the 2,623-token timing shape.
- **The real-length figure is an estimate.** The ≈811 ms/step weights two fixed-length synthetic runs by a bucket distribution simulated from mix665k lengths. It is not a real-data run, the baseline was not measured at those lengths, and the length-measurement script is not part of this branch.
- **The all-reduce bottleneck is unresolved** (about 21% of device time), and the three attempts above failed.
- **1.70× is chained across two A/B/A batches.** Each step passed its own A/B/A; the end-to-end ratio uses the first batch's baseline and the second batch's candidate.
- **Scope of the splice.** It supports one `<image>` per sample with right padding, and the fast path additionally needs a batch-shared image position. Anything else raises rather than diverging silently. Text-only batches do not go through the splice.
- **Different library versions from the GPU recipe** (transformers 4.56.2 instead of 4.37.2; no DeepSpeed, flash-attn, Apex or Liger). No TPU-versus-GPU comparison is made.

## Source and reproduction

Scripts, CPU tests, exact flags and environment variables are in [`tpu_smoke/`](../tpu_smoke/README.md). Baseline and adopted configuration differ only in `--attn sdpa` versus `--attn splash --splice-concat` with `SPLASH_PAD_TO=512`.
