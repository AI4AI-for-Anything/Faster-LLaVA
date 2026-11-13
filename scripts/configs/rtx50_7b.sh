# RTX 50 series (RoundPipe) training configuration.
export DATA_WORKERS=${DATA_WORKERS:-16}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONFAULTHANDLER=1
# Measured on 8x RTX 5080 (docs/rtx50-results.md): a fixed stage plan with a light first stage,
# 2 x GPUs + 1 token-balanced microbatches (NUM_MICROBATCH=0), and fewer kernel launches per layer.
export STAGE_PLAN=${STAGE_PLAN:-3,1,2}
export NUM_MICROBATCH=${NUM_MICROBATCH:-0}
export LLAVA_RTX50_FUSED_RMSNORM=${LLAVA_RTX50_FUSED_RMSNORM:-1}
