#!/usr/bin/env bash
# LLaVA-1.5-7B stage-two training on one host of RTX 50 series GPUs (16 GB each), via RoundPipe.
set -euo pipefail
: "${LLAVA_ASSETS:?Set LLAVA_ASSETS to the prepared public data/model directory}"
: "${LLAVA_IMAGES:?Set LLAVA_IMAGES to the image root}"
: "${LLAVA_OUTPUT:?Set LLAVA_OUTPUT to a new output directory}"
# One optimizer step holds MICROBATCH x REFERENCE_GPUS x GRAD_ACCUM samples, grouped as the
# DeepSpeed recipe on REFERENCE_GPUS GPUs groups them. RoundPipe uses every visible GPU.
REFERENCE_GPUS=${REFERENCE_GPUS:-8}
MICROBATCH=${MICROBATCH:-16}
GRAD_ACCUM=${GRAD_ACCUM:-1}
for value in "$REFERENCE_GPUS" "$MICROBATCH" "$GRAD_ACCUM"; do
  [[ "$value" =~ ^[1-9][0-9]*$ ]] || { echo 'Batch/GPU values must be positive integers.' >&2; exit 2; }
done
(( REFERENCE_GPUS * MICROBATCH * GRAD_ACCUM == 128 )) || { echo 'Global batch must remain 128.' >&2; exit 2; }
[[ ! -e "$LLAVA_OUTPUT" || -n "${RESUME:-}" ]] || { echo 'Use a new output directory (or RESUME=1 to continue it).' >&2; exit 2; }
REPO_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$REPO_ROOT"
source scripts/configs/rtx50_7b.sh
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false
python llava/train/train_rtx50.py \
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
  --per_device_train_batch_size "$MICROBATCH" --reference_world_size "$REFERENCE_GPUS" \
  --gradient_accumulation_steps "$GRAD_ACCUM" \
  --stage_plan "${STAGE_PLAN}" --num_microbatch "${NUM_MICROBATCH}" \
  --save_strategy steps --save_steps 1000 --save_total_limit 1 \
  --learning_rate 2e-5 --weight_decay 0. --warmup_ratio 0.03 --lr_scheduler_type cosine \
  --max_grad_norm 0 \
  --logging_steps 1 --tf32 True --model_max_length 2048 \
  --dataloader_num_workers "${DATA_WORKERS}" --lazy_preprocess True --report_to none \
  "$@"
