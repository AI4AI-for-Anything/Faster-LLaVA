# Training and reproduction

Use [train_faster_7b.sh](../scripts/train_faster_7b.sh) as the single entry point for optimized LLaVA-1.5-7B stage-two training. It loads the [optimization configuration](../scripts/configs/faster_7b.sh), accepts configurable paths and GPU/batch counts, and checks that the global batch is 128 and the output directory is new. Its training arguments follow the measured H20 recipe; this portable launcher has not been used for a new full training run.

For other training stages or LoRA examples, refer to the [upstream training scripts](https://github.com/haotian-liu/LLaVA/tree/main/scripts). The scripts under `scripts/v1_5/eval/` are upstream evaluation examples.

## Environment

Use Linux with NVIDIA CUDA. The measured training environment used Python 3.10, PyTorch **2.1.2+cu121**, torchvision 0.16.2+cu121, Transformers **4.37.2**, NumPy 1.26.4, DeepSpeed 0.12.6, Accelerate 0.21.0, FlashAttention 2.5.8, Triton 2.1.0, NVIDIA Apex (package version 0.1) and Liger Kernel 0.4.2. Apex's version string alone does not pin its source revision.

The repository's `pyproject.toml` pins the main training dependencies. In an isolated environment, install the project with its training extras (`pip install -e '.[train]'`), then install compatible FlashAttention, Liger Kernel and NVIDIA Apex CUDA extensions. Follow [FlashAttention](https://github.com/Dao-AILab/flash-attention) and [NVIDIA Apex](https://github.com/NVIDIA/apex) build instructions for your CUDA toolkit. The unqualified PyPI package named `apex` is not a replacement for NVIDIA Apex. Check these imports before launching:

```bash
python -c 'import torch, flash_attn, deepspeed; from apex.normalization import FusedRMSNorm; from liger_kernel.ops.swiglu import LigerSiLUMulFunction; print(torch.__version__, torch.cuda.device_count())'
```

The optimized kernels target the versions above. A fresh environment installation is not validated by the archived results.

## Public data and weights

Obtain the files from the [official visual instruction tuning instructions](https://github.com/haotian-liu/LLaVA#visual-instruction-tuning):

- [Vicuna-7B v1.5](https://huggingface.co/lmsys/vicuna-7b-v1.5)
- [CLIP ViT-L/14 at 336 pixels](https://huggingface.co/openai/clip-vit-large-patch14-336)
- [Official 7B stage-one projector](https://huggingface.co/liuhaotian/llava-v1.5-mlp2x-336px-pretrain-vicuna-7b-v1.5)
- [Instruction JSON](https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K/blob/main/llava_v1_5_mix665k.json) and the prescribed COCO, GQA, OCR-VQA, TextVQA and Visual Genome images.

Expected paths:

```text
$LLAVA_ASSETS/
  models/vicuna-7b-v1.5/
  models/clip-vit-large-patch14-336/
  models/llava-v1.5-mlp2x-336px-pretrain-vicuna-7b-v1.5/mm_projector.bin
  data/llava_v1_5_mix665k.json
$LLAVA_IMAGES/
  coco/train2017/
  gqa/images/
  ocr_vqa/images/
  textvqa/train_images/
  vg/VG_100K/
  vg/VG_100K_2/
```

The H20 training data contained 665,298 records and 349,034 unique images, with no missing training images. No dataset images or checkpoint weights are distributed here.

The source official fine-tuning script defaults to **13B**. This run substitutes the matching 7B base and projector, preserving the main recipe: one epoch, global batch 128, learning rate 2e-5, cosine schedule, warmup ratio 0.03, weight decay 0, maximum length 2,048, BF16/TF32 and ZeRO-3. CLIP is frozen; the language model and projector are trained without LoRA. This is full stage-two fine-tuning, not continuation of the final released LLaVA checkpoint.

Before a long run, validate checkpoint saving. The H20 run needed the local Vicuna `generation_config.json` to omit `temperature` and `top_p` while `do_sample` was false, because Transformers 4.37.2 rejects that combination on save. The original file was backed up. Training does not call `generate`; evaluation explicitly uses greedy decoding. See the [H20 results and evaluation protocol](h20-results.md).

## Launch

From the repository root, in the prepared environment:

```bash
export LLAVA_ASSETS=/path/to/assets
export LLAVA_IMAGES=/path/to/images
export LLAVA_OUTPUT=/path/to/new-output
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
bash scripts/train_faster_7b.sh
```

On a shared host, reserve the GPUs through the local scheduler or locking mechanism before launching. Do not compete with another process on the same devices.

## Memory

**Agent hint:** Before any trial run, inspect GPU count and available memory, then choose `MICROBATCH` and `GRAD_ACCUM` accordingly. For a tighter memory budget, first reduce the microbatch and increase gradient accumulation, keeping `NPROC × MICROBATCH × GRAD_ACCUM = 128`. For example, `NPROC=8 MICROBATCH=8 GRAD_ACCUM=2` preserves the global batch. Increasing accumulation alone does not reduce memory per microbatch.

After adjusting the configuration, run a short forward/backward and checkpoint-save smoke test, and measure peak memory and step time. If this still does not fit, consider gradient checkpointing or fewer retained activation layers; check implementation/configuration support and validate the change before a full run.
