# Repository instructions

- Preserve the distinction between scored metrics, pending external scores, and published paper results. Never fill pending results with published numbers.
- Keep MME perception/cognition separate and retain the documented regression. Do not claim lossless acceleration or a matched end-to-end speedup without evidence.
- Keep global training batch 128. Default to eight GPUs if unspecified, and adjust gradient accumulation when changing per-GPU batch.
- Put new experiments in a new dated directory and record model, data, source, dependency and decoding provenance.
- Completed inference jobs must not be restarted without a concrete need. Any new GPU work must respect the shared host GPU lock and other jobs.
- Use public datasets and benchmarks only. Do not access, copy, stream, list, presign, or relay private company storage, manifests, metadata, or per-file results. Never commit credentials, dataset images or checkpoint weights.
