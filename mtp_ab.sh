#!/usr/bin/env bash                                                                                                                                        
set -euo pipefail
MODEL=Qwen/Qwen3.8-27B-FP8
PORT=8000
K=${K:-2}                     # num_speculative_tokens                                                                                                     
COMMON=(--max-model-len 50000 --kv-cache-dtype fp8 --reasoning-parser qwen3
        --limit-mm-per-prompt '{"image":0,"video":0}' --port $PORT
        --gpu-memory-utilization 0.90)

launch() {  # $1 = tag, rest = extra args                                                                                                                  
  tag=$1; shift
  vllm serve $MODEL "${COMMON[@]}" "$@" > "serve_${tag}.log" 2>&1 &
  PID=$!
  for i in $(seq 1 180); do
    curl -sf localhost:$PORT/health >/dev/null 2>&1 && return
    kill -0 $PID 2>/dev/null || { echo "server $tag died, see serve_${tag}.log"; exit 1; }
    sleep 5
  done
  echo "timeout waiting for $tag"; exit 1
}
stop() { kill $PID 2>/dev/null || true; wait $PID 2>/dev/null || true; sleep 3; }

launch base
for c in 1 8 32; do python mtp_bench.py http://localhost:$PORT $c base; done
stop

launch mtp --speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":$K}"
grep -m1 -o 'SpeculativeConfig([^)]*' serve_mtp.log || echo "WARN: no SpeculativeConfig in log"
for c in 1 8 32; do python mtp_bench.py http://localhost:$PORT $c mtp; done
stop

echo "== greedy output diff (must be empty) =="
diff out_base_c1.txt out_mtp_c1.txt && echo IDENTICAL
echo "== summary =="; cat results.tsv

