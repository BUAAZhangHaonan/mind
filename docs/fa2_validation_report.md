# FA2 validation report

| Model | eager load | eager fwd | eager VRAM | FA2 load | FA2 fwd | FA2 VRAM | Notes |
|---------------------|------------|-----------|------------|----------|---------|----------|-------|
| Qwen3-VL-8B | FAIL | FAIL | 16.51 GB | FAIL | NA | 0.02 GB | Timeout after 90s; FA2: Timeout after 90s |
| InternVL3.5-8B | PASS | PASS | 17.57 GB | FAIL | NA | 0.02 GB | FA2: ImportError: FlashAttention2 has been toggled on, but it cannot be used due to the following error: the package flash_attn seems to be not installed. Please refer to the documentation of https://huggingface.co/docs/transformers/perf_infer_gpu_one#flashattention-2 to install Flash Attention 2. |
| LLaVA-OneVision-7B | PASS | PASS | 16.22 GB | FAIL | NA | 0.02 GB | FA2: ImportError: FlashAttention2 has been toggled on, but it cannot be used due to the following error: the package flash_attn seems to be not installed. Please refer to the documentation of https://huggingface.co/docs/transformers/perf_infer_gpu_one#flashattention-2 to install Flash Attention 2. |
| Molmo-7B-D-0924 | PASS | PASS | 15.47 GB | FAIL | NA | 0.02 GB | FA2: ValueError: MolmoForCausalLM does not support Flash Attention 2.0 yet. Please request to add support where the model is hosted, on its model hub page: https://huggingface.co/allenai/Molmo-7B-D-0924/discussions/new or in the Transformers GitHub repo: https://github.com/huggingface/transformers/issues/new |
