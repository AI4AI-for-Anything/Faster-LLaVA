# Files added for RTX 50 series GPUs

The RTX 50 recipe adds **4 source and launcher files** and touches no existing training code: the H20 path, its defaults and `pyproject.toml` are unchanged. Documentation, figures and `requirements-rtx50.txt` are listed separately below.

| File | Area | Change |
|---|---|---|
| [llava/model/rtx50_sequential.py](../../llava/model/rtx50_sequential.py) | Model view | Views `LlavaLlamaForCausalLM` as the `nn.Sequential` RoundPipe needs (vision prefix, packed decoder layers, supervised-row head and loss) without copying weights; packs each microbatch on the host; optional fused RMSNorm (`LLAVA_RTX50_FUSED_RMSNORM`). |
| [llava/train/train_rtx50.py](../../llava/train/train_rtx50.py) | Training loop | Single-process trainer: the bf16 full-finetune setup of `train.train()`, the reference sampler's step membership, token-balanced microbatches, a fixed or automatic RoundPipe stage plan, synchronous AdamW with fp32 master weights, cosine schedule, checkpoint and resume in the upstream format. |
| [scripts/train_rtx50_7b.sh](../../scripts/train_rtx50_7b.sh) | Launcher | Same arguments as `train_faster_7b.sh`; checks global batch 128 and a new output directory (or `RESUME=1`). |
| [scripts/configs/rtx50_7b.sh](../../scripts/configs/rtx50_7b.sh) | Configuration | The measured settings: stage plan `3,1,2`, 2 × GPUs + 1 microbatches, fused RMSNorm. |

Supporting files: [requirements-rtx50.txt](../../requirements-rtx50.txt) (environment), [README_5080.md](../../README_5080.md), [rtx50-results.md](../rtx50-results.md), and `docs/performance-views/03-rtx50-benchmark.svg`, `04-rtx50-history.svg`, `rtx50-history.json` (`python docs/performance-views/render.py rtx50`).

The recipe reuses `make_supervised_data_module` and `get_modality_length_grouped_indices`, reproduces the token layout of `LlavaMetaForCausalLM.prepare_inputs_labels_for_multimodal`, and writes checkpoints that load with the upstream evaluation loader (`load_pretrained_model`; as upstream, the model path must contain "llava"). Packing, global supervised-token loss normalisation and RoundPipe's bf16 gradient accumulation change the training trajectory; see [rtx50-results.md](../rtx50-results.md#numerical-checks).
