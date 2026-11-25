#!/usr/bin/env bash
# Runs ON the pod. One serve attempt: start vllm serve, wait for ready or
# death, then send one text prompt. Usage: try.sh <tag> [extra serve args...]
set -uo pipefail
tag=$1; shift
cd /root/work/vllm
source .venv/bin/activate
export HF_HOME=/root/hf
log=/root/logs/serve_$tag.log
mkdir -p /root/logs
pkill -f "vllm serve" 2>/dev/null; sleep 2
t0=$(date +%s)
vllm serve thinkingmachines/Inkling-2-Mini \
  --enforce-eager --max-model-len 4096 --max-num-batched-tokens 1024 \
  --max-num-seqs 4 --gpu-memory-utilization 0.92 --port 8000 "$@" \
  >"$log" 2>&1 &
pid=$!
while kill -0 $pid 2>/dev/null; do
  grep -q "Application startup complete" "$log" && break
  (( $(date +%s) - t0 > 900 )) && { echo "TIMEOUT"; break; }
  sleep 3
done
echo "elapsed $(( $(date +%s) - t0 ))s"
if ! kill -0 $pid 2>/dev/null; then
  echo "=== SERVER DIED ==="
  grep -n -E "Error|error|Exception|Traceback" "$log" | head -40
  echo "--- tail ---"; tail -60 "$log"
  exit 1
fi
grep -E "Available KV cache|Maximum concurrency|Model loading took|GiB" "$log" | head
curl -s localhost:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "thinkingmachines/Inkling-2-Mini",
  "messages": [{"role": "user", "content": "What is the capital of France? Answer in one sentence."}],
  "max_tokens": 64, "temperature": 0}' | head -c 2000
echo
if ! kill -0 $pid 2>/dev/null; then
  echo "=== SERVER DIED DURING REQUEST ==="; tail -60 "$log"; exit 1
fi
echo "server still up (pid $pid)"
