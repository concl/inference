# inference

## Command
```bash
MODEL=Qwen/Qwen3.8-27B

CUDA_VISIBLE_DEVICES=0,1,2,3 \
.venv/bin/vllm serve "$MODEL" \
  --served-model-name Qwen3.8-27B \
  --tensor-parallel-size 4 \
  --dtype bfloat16 \
  --gpu-memory-utilization 0.90 \
  --max-model-len 32768 \
  --max-num-seqs 32 \
  --host 0.0.0.0 \
  --port 8000
  --reasoning-parser qwen3
```
