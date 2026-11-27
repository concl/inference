#!/usr/bin/env bash
# Launch NIXL PD disaggregation: prefill (GPU 0) + decode (GPU 1) + proxy (8300).
#
#   bash experiments/pd_disagg/run.sh                 # MTP off (default)
#   PD_MTP=1 bash experiments/pd_disagg/run.sh        # MTP n=3 (unverified)
#
# Env:
#   GPU_PREFILL / GPU_DECODE   default 0 / 1
#   PD_MTP                     0 = off, 1 = MTP n=3 on both sides
#   FORCE                      1 = skip the free-GPU check
set -uo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
EXP="$ROOT/experiments/pd_disagg"
INF="$ROOT/.venv/bin/inference"
PY="$ROOT/.venv/bin/python"
MODEL="/home/kaleox.dev/models-cache/qwen3.8-27b-int4-autoround"
GPU_PREFILL="${GPU_PREFILL:-0}"
GPU_DECODE="${GPU_DECODE:-1}"
PD_MTP="${PD_MTP:-0}"
P_PORT=8100
D_PORT=8200
PROXY_PORT=8300
OUT="${OUT:-$EXP/artifacts/pd-$(date -u +%Y%m%dT%H%M%SZ)}"
PROFILE="${PROFILE:-0}"
PROFILE_ONLY="${PROFILE_ONLY:-0}"

P_EXTRA=()
D_EXTRA=()
if [[ "$PROFILE" == "1" ]]; then
  mkdir -p "$OUT/prefill/torch" "$OUT/decode/torch"
  _prof() {
    printf '{"profiler":"torch","ignore_frontend":true,"torch_profiler_dir":"%s","torch_profiler_with_stack":false,"torch_profiler_record_shapes":true,"torch_profiler_with_memory":false,"torch_profiler_use_gzip":false}' "$1"
  }
  P_EXTRA=(--profiler-config "$(_prof "$OUT/prefill/torch")")
  D_EXTRA=(--profiler-config "$(_prof "$OUT/decode/torch")")
fi

[[ -f "$MODEL/config.json" ]] || { echo "missing model: $MODEL" >&2; exit 1; }
"$PY" -c 'import nixl' 2>/dev/null || { echo "nixl is not installed (uv pip install nixl==1.5.0)" >&2; exit 1; }
[[ "$GPU_PREFILL" != "$GPU_DECODE" ]] || { echo "prefill and decode need different GPUs" >&2; exit 1; }

if [[ "${FORCE:-0}" != "1" ]]; then
  for label in prefill decode; do
    gpu="$GPU_PREFILL"; [[ "$label" == "decode" ]] && gpu="$GPU_DECODE"
    used=$(nvidia-smi -i "$gpu" --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null || echo 99999)
    if [[ "$used" -gt 2048 ]]; then
      echo "GPU $gpu ($label) already has ${used} MiB in use; free it or set FORCE=1" >&2
      exit 1
    fi
  done
fi

if [[ "$PD_MTP" == "1" ]]; then
  P_CFG="$EXP/prefill-mtp.yaml"
  D_CFG="$EXP/decode-mtp.yaml"
  echo "[pd] MTP n=3 (UNVERIFIED on hybrid SSM + PD)"
else
  P_CFG="$EXP/prefill.yaml"
  D_CFG="$EXP/decode.yaml"
fi

for p in "$P_PORT" "$D_PORT" "$PROXY_PORT"; do
  curl -sf -m 1 "http://127.0.0.1:$p/v1/models" >/dev/null 2>&1 && { echo "port $p already serving" >&2; exit 1; }
done

mkdir -p "$OUT"
echo "[pd] run dir: $OUT"
{ date -Is; nvidia-smi; } > "$OUT/host.txt" 2>&1 || true
"$PY" -c 'import importlib.metadata as m,json; print(json.dumps({n:m.version(n) for n in ["vllm","torch","nixl"]},indent=2))' > "$OUT/environment.json" 2>/dev/null || true

P_PID=""
D_PID=""
X_PID=""
cleanup() {
  for pid in "$X_PID" "$D_PID" "$P_PID"; do
    [[ -n "$pid" ]] || continue
    if kill -0 "$pid" 2>/dev/null; then
      kill -TERM "$pid" 2>/dev/null || true
      for _ in $(seq 1 20); do kill -0 "$pid" 2>/dev/null || break; sleep 1; done
      kill -KILL "$pid" 2>/dev/null || true
      wait "$pid" 2>/dev/null || true
    fi
  done
}
trap cleanup EXIT INT TERM

echo "[pd] prefill on GPU $GPU_PREFILL :$P_PORT"
env CUDA_VISIBLE_DEVICES="$GPU_PREFILL" \
    VLLM_PORT=20000 \
    VLLM_NIXL_SIDE_CHANNEL_PORT=5600 \
    VLLM_KV_CACHE_LAYOUT=HND \
    UCX_NET_DEVICES=all \
    UCX_TLS=all \
    OPENAI_API_KEY=pd-local \
    PYTHONPATH="$EXP/no_mm_warmup${PYTHONPATH:+:$PYTHONPATH}" \
    VLLM_SSM_CONV_STATE_LAYOUT=DS \
  "$INF" serve --config "$P_CFG" --gpus "$GPU_PREFILL" "${P_EXTRA[@]}" > "$OUT/prefill.log" 2>&1 &
P_PID=$!

echo "[pd] decode on GPU $GPU_DECODE :$D_PORT"
env CUDA_VISIBLE_DEVICES="$GPU_DECODE" \
    VLLM_PORT=30000 \
    VLLM_NIXL_SIDE_CHANNEL_PORT=5601 \
    VLLM_KV_CACHE_LAYOUT=HND \
    UCX_NET_DEVICES=all \
    UCX_TLS=all \
    OPENAI_API_KEY=pd-local \
    PYTHONPATH="$EXP/no_mm_warmup${PYTHONPATH:+:$PYTHONPATH}" \
    VLLM_SSM_CONV_STATE_LAYOUT=DS \
  "$INF" serve --config "$D_CFG" --gpus "$GPU_DECODE" "${D_EXTRA[@]}" > "$OUT/decode.log" 2>&1 &
D_PID=$!

wait_models() {
  local port=$1 name=$2 pid=$3
  for _ in $(seq 1 600); do
    curl -sf -m 2 "http://127.0.0.1:$port/v1/models" >/dev/null 2>&1 && return 0
    kill -0 "$pid" 2>/dev/null || { echo "[pd] $name died; tail:" >&2; tail -n 40 "$OUT/$name.log" >&2; return 1; }
    sleep 2
  done
  echo "[pd] $name not ready" >&2
  tail -n 40 "$OUT/$name.log" >&2
  return 1
}

wait_models "$P_PORT" prefill "$P_PID" || exit 1
wait_models "$D_PORT" decode "$D_PID" || exit 1
echo "[pd] both servers ready"

"$PY" "$EXP/proxy.py" --host 127.0.0.1 --port "$PROXY_PORT" \
  --prefiller-hosts 127.0.0.1 --prefiller-ports "$P_PORT" \
  --decoder-hosts 127.0.0.1 --decoder-ports "$D_PORT" \
  > "$OUT/proxy.log" 2>&1 &
X_PID=$!

for _ in $(seq 1 60); do
  curl -sf -m 2 "http://127.0.0.1:$PROXY_PORT/healthcheck" >/dev/null 2>&1 && break
  sleep 1
done
curl -sf -m 2 "http://127.0.0.1:$PROXY_PORT/healthcheck" >/dev/null 2>&1 || { echo "[pd] proxy not ready" >&2; tail -n 40 "$OUT/proxy.log" >&2; exit 1; }
echo "[pd] proxy ready on :$PROXY_PORT"

if [[ "$PROFILE" == "1" ]]; then
  echo "[pd] warmup request (unprofiled)"
  "$INF" bench --config "$EXP/bench-vision-arena.yaml" --result-dir "$OUT" \
    --num-prompts 1 --num-warmups 0 --result-filename warmup.json \
    > "$OUT/warmup.log" 2>&1 || echo "[pd] warmup failed" >&2

  echo "[pd] start_profile on both servers"
  curl -fsS -m 10 -X POST "http://127.0.0.1:$P_PORT/start_profile" > "$OUT/prefill-profile-start.txt" 2>&1 || \
    echo "[pd] prefill start_profile failed" >&2
  curl -fsS -m 10 -X POST "http://127.0.0.1:$D_PORT/start_profile" > "$OUT/decode-profile-start.txt" 2>&1 || \
    echo "[pd] decode start_profile failed" >&2

  echo "[pd] profiled request(s)"
  "$INF" bench --config "$EXP/bench-vision-arena.yaml" --result-dir "$OUT" \
    --num-prompts 2 --num-warmups 0 --result-filename bench-trace.json \
    > "$OUT/bench-trace.log" 2>&1 || echo "[pd] trace bench failed" >&2

  echo "[pd] stop_profile (flush)"
  curl -fsS -m 900 -X POST "http://127.0.0.1:$P_PORT/stop_profile" > "$OUT/prefill-profile-stop.txt" 2>&1 || \
    echo "[pd] prefill stop_profile failed" >&2
  curl -fsS -m 900 -X POST "http://127.0.0.1:$D_PORT/stop_profile" > "$OUT/decode-profile-stop.txt" 2>&1 || \
    echo "[pd] decode stop_profile failed" >&2

  for _ in $(seq 1 90); do
    p=$(find "$OUT/prefill/torch" -name '*.pt.trace.json' -size +1k 2>/dev/null | wc -l)
    d=$(find "$OUT/decode/torch" -name '*.pt.trace.json' -size +1k 2>/dev/null | wc -l)
    [[ "$p" -ge 1 && "$d" -ge 1 ]] && break
    sleep 2
  done
  echo "[pd] traces:"
  find "$OUT/prefill/torch" "$OUT/decode/torch" -name '*.pt.trace.json' -printf '%s %p\n' 2>/dev/null || true
fi

if [[ "$PROFILE_ONLY" != "1" ]]; then
  for ds in vision-arena; do
    echo "[pd] bench $ds"
    "$INF" bench --config "$EXP/bench-$ds.yaml" --result-dir "$OUT" > "$OUT/bench-$ds.log" 2>&1 || \
      echo "[pd] bench $ds failed (see $OUT/bench-$ds.log)" >&2
  done
fi

echo "[pd] done: $OUT"
find "$OUT" -maxdepth 1 -type f | sort