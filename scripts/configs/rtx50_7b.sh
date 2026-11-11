# RTX 50 series (RoundPipe) training configuration.
export DATA_WORKERS=${DATA_WORKERS:-16}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONFAULTHANDLER=1
