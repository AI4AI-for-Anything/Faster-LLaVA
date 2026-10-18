# Files changed

The optimizations span **12 source/configuration files** (8 added, 4 modified). This count excludes later documentation, launchers and evaluation archives.

| File | Area | Change |
|---|---|---|
| [llava/train/train.py](../../llava/train/train.py) | Data & sampling | Cache token/label lengths and pass normalization metadata through the data pipeline. |
| [llava/train/llava_trainer.py](../../llava/train/llava_trainer.py) | Data & sampling | Group by token length; balance all buckets in an optimizer step; normalize loss by global supervised-token count. |
| [llava/model/language_model/llava_llama.py](../../llava/model/language_model/llava_llama.py) | Model integration | Enable the optimization switches, fused normalization and selective activation retention. |
| [llava/model/unpad_mlp.py](../../llava/model/unpad_mlp.py) | Packed computation | Run the MLP on valid token rows. |
| [llava/model/unpad_attention.py](../../llava/model/unpad_attention.py) | Packed computation | Remove padding from attention projections. |
| [llava/model/packed_attention.py](../../llava/model/packed_attention.py) | Packed computation | Keep attention projections, RoPE and variable-length attention in a packed layout. |
| [llava/model/packed_decoder.py](../../llava/model/packed_decoder.py) | Packed computation | Carry packed tokens and sequence metadata across decoder layers. |
| [llava/model/packed_rope.py](../../llava/model/packed_rope.py) | Packed computation | Apply Liger rotary embeddings to packed queries and keys. |
| [llava/model/recompute_swiglu.py](../../llava/model/recompute_swiglu.py) | Activation & loss memory | Recompute SwiGLU activations during backward to reduce saved tensors. |
| [llava/model/fused_head_loss.py](../../llava/model/fused_head_loss.py) | Activation & loss memory | Compute output projection and cross-entropy for supervised rows; optional Liger CE. |
| [llava/model/chunked_head.py](../../llava/model/chunked_head.py) | Activation & loss memory | Optional chunked head implementation; the default preset uses chunk size 0 (unchunked). |
| [scripts/zero3.json](../../scripts/zero3.json) | Communication | Keep parameters resident under ZeRO-3 with persistence threshold 1e9. |

The [optimization preset](../../scripts/configs/faster_7b.sh) selects the measured configuration. Some earlier implementations remain as optional paths; the packed decoder supersedes parts of the earlier gather/scatter path. The presence of a file does not imply an independent speedup.

Token repartitioning, global-token loss normalization and numerical kernels change the training trajectory. The completed model also regresses on MME; see [full-training results](../h20-results.md).
