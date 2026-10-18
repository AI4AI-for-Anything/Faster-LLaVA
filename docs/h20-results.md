# H20 training and evaluation results

H20 training and evaluation results for optimized LLaVA-1.5-7B.

The table below presents **nine scored benchmark comparisons**. These results do **not** establish lossless acceleration.

| Benchmark ↑ | Official checkpoint, same-protocol rerun | Our checkpoint |
|---|---:|---:|
| GQA testdev balanced | 61.91 | 62.35 |
| VizWiz test | 50.42 | 51.89 |
| ScienceQA-IMG | 67.82 | 68.52 |
| TextVQA val 0.5.1 | 58.23 | 58.18 |
| POPE macro F1 | 85.90 | 85.92 |
| MMBench EN circular | 64.29 | 65.56 |
| MMBench CN circular | 58.08 | 61.51 |
| SEED full (19242) | 58.67 | 59.21 |
| MM-Vet v1 | 33.24 | 33.85 |

Our complete stage-two fine-tuning took **7 h 14 min on 8×H20** (26,041.25 seconds, 5,198 steps). A matched full-run H20 baseline is not established by this report, so no end-to-end speedup ratio is claimed here.

Protocol notes:

- ScienceQA uses the original 2023 source with PyTorch 2.0.1 / Transformers 4.31. Five protocols × two checkpoints were checked. Preprocessing explains part of the discrepancy; the official rerun remains 1.0235 points above the published 66.8, with the remainder unexplained.
- SEED uses all 19,242 historical questions, including 5,009 video questions.
- VizWiz uses the released 8,000-question full test, not the historical 4,000-question test-dev.
- MM-Vet uses three runs per checkpoint with the same `gpt-6-luna-2026-09-22` judge. These numbers are not directly comparable with the paper's GPT-4 scores.
- MMBench uses verified local circular scoring. External server submission is separate.

## Training setting

Full second-stage instruction fine-tuning starts from Vicuna plus the official stage-one projector; it is not continued training of the released final LLaVA checkpoint. Stage one was not retrained.

| Setting | Value |
|---|---|
| Training records | 665,298; 1 epoch |
| Global batch | 128 = 8 GPUs × 16 examples/GPU × accumulation 1 |
| Learning rate | 2e-5 |
| Maximum sequence length | 2,048 |
| Precision / optimizer partitioning | BF16, TF32, ZeRO-3 |
| Trainable components | Language model and projector; frozen CLIP; no LoRA |

Token grouping/global-step balancing, global-token loss normalization, and packed/fused/BF16 operations change the numerical trajectory. Matching data and primary hyperparameters does not imply strict equivalence. For another GPU count or memory limit, preserve global batch 128 with gradient accumulation; use eight GPUs as the default planning assumption.

## Source and reproduction

The original [upstream README](../README.upstream.md) and Apache-2.0 [license](../LICENSE) are retained.

See the [training guide](training.md), [training launcher](../scripts/train_faster_7b.sh) and [optimization configuration](../scripts/configs/faster_7b.sh).
