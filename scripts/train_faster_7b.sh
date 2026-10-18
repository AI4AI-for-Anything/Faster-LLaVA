#!/usr/bin/env bash
# Configurable entry point for optimized LLaVA-1.5-7B stage-two training.
set -euo pipefail
: "${LLAVA_ASSETS:?Set LLAVA_ASSETS to the prepared public data/model directory}"
: "${LLAVA_IMAGES:?Set LLAVA_IMAGES to the image root}"
: "${LLAVA_OUTPUT:?Set LLAVA_OUTPUT to a new output directory}"
NPROC=${NPROC:-8}
MICROBATCH=${MICROBATCH:-16}
GRAD_ACCUM=${GRAD_ACCUM:-1}
for value in "$NPROC" "$MICROBATCH" "$GRAD_ACCUM"; do
  [[ "$value" =~ ^[1-9][0-9]*$ ]] || { echo 'Batch/GPU values must be positive integers.' >&2; exit 2; }
done
(( NPROC * MICROBATCH * GRAD_ACCUM == 128 )) || { echo 'Global batch must remain 128.' >&2; exit 2; }
[[ ! -e "$LLAVA_OUTPUT" ]] || { echo 'Use a new output directory.' >&2; exit 2; }
REPO_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$REPO_ROOT"
source scripts/configs/faster_7b.sh
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false
torchrun --standalone --nnodes=1 --nproc-per-node="$NPROC" \
  llava/train/train_mem.py \
  --deepspeed ./scripts/zero3.json \
  --model_name_or_path "${LLAVA_ASSETS}/models/vicuna-7b-v1.5" \
  --version v1 \
  --data_path "${LLAVA_ASSETS}/data/llava_v1_5_mix665k.json" \
  --image_folder "${LLAVA_IMAGES}" \
  --vision_tower "${LLAVA_ASSETS}/models/clip-vit-large-patch14-336" \
  --pretrain_mm_mlp_adapter "${LLAVA_ASSETS}/models/llava-v1.5-mlp2x-336px-pretrain-vicuna-7b-v1.5/mm_projector.bin" \
  --mm_projector_type mlp2x_gelu --mm_vision_select_layer -2 \
  --mm_use_im_start_end False --mm_use_im_patch_token False \
  --image_aspect_ratio pad --group_by_modality_length True \
  --bf16 True --output_dir "${LLAVA_OUTPUT}" \
  --num_train_epochs 1 \
  --per_device_train_batch_size "$MICROBATCH" --per_device_eval_batch_size 4 \
  --gradient_accumulation_steps "$GRAD_ACCUM" \
  --evaluation_strategy no \
  --save_strategy steps --save_steps 1000 --save_total_limit 1 \
  --learning_rate 2e-5 --weight_decay 0. --warmup_ratio 0.03 --lr_scheduler_type cosine \
  --logging_steps 1 --tf32 True --model_max_length 2048 \
  --gradient_checkpointing True --dataloader_num_workers 4 --lazy_preprocess True --report_to none
