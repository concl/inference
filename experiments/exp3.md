# exp3 — prefix caching on, BF16, vLLM#48375 patch applied

## Server

Started 2026-10-01 06:00, still running.

```bash
vllm serve Qwen/Qwen3.8-27B \
  --served-model-name Qwen3.8-27B \
  --tensor-parallel-size 4 \
  --dtype bfloat16 \
  --gpu-memory-utilization 0.95 \
  --max-model-len 262144 \
  --max-num-seqs 32 \
  --host 127.0.0.1 --port 8000 \
  --reasoning-parser qwen3 \
  --enable-auto-tool-choice --tool-call-parser qwen3_xml
```

| Setting | Value |
|---|---|
| Model | `Qwen/Qwen3.8-27B` (BF16) |
| GPUs / TP | `0,1,2,3` / 4 |
| gpu-memory-utilization | 0.95 |
| max-model-len | 262144 |
| max-num-seqs | 32 |
| prefix caching | on (vLLM default, no `--no-enable-prefix-caching`) |
| patch | `experiments/vllm-48375-mamba-drop-eagle-block.patch`, applied to `.venv/.../site-packages` |

## Bench

Run 2026-10-01, 20 prompts each, `--max-concurrency 2`, `--request-rate inf`.
Full commands in `README.md`. Output JSON in `experiments/results/`.

## Results

| Metric | ShareGPT | BFCL |
|---|---|---|
| TTFT P50 / P95 (ms) | 405 / 907 | 458 / 778 |
| TPOT P50 / P95 (ms) | 53.1 / 57.1 | 56.2 / 57.1 |
| ITL P50 / P95 (ms) | 52.4 / 56.8 | 54.0 / 109.5 |
| Output tok/s | 35.4 | 32.9 |
| Total tok/s | 110.0 | 126.2 |
| Duration (s) | 72.4 | 105.3 |
| Prompts completed / failed | 20 / 0 | 20 / 0 |

## Notes

- These numbers are from the **BF16** server above, not the INT8 one in `exp2.md`.
  The bench JSON does not record which server answered: `--model Qwen3.8-27B` is
  the served name both servers use, and `--tokenizer` points at the INT8 repo for
  token counting only. Confirm from the server, not the JSON.
- BFCL ITL P95 (109.5 ms) is ~2x ShareGPT's. Tool-schema requests are the likely
  cause; unverified.
