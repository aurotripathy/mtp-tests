# MTP benchmark

A/B of speculative decoding on `Qwen/Qwen3.8-27B-FP8` under vLLM 0.32. The checkpoint ships multi-token prediction (MTP) heads; vLLM turns them on with `"method": "mtp"`.

`mtp_bench.py` fires the same eight prompts at concurrency 1, 8, and 32 (`temperature=0`, `seed=0`, thinking off) and records aggregate throughput, per-request decode speed, time to first token, and draft acceptance from `/metrics`. `mtp_ab.sh` serves a spec-off baseline, then the same model with MTP, and diffs the greedy conc=1 outputs.

`MAX_TOK = 512` is sent as `max_tokens`. That caps **completion** tokens. Prompt length is whatever each seed tokenizes to, about 20–40 tokens, and is not limited by this setting. `out=512` at conc=1 means the request hit the cap. `out` below `512 × concurrency` means some requests stopped at EOS first.

## Serve

```bash
vllm serve Qwen/Qwen3.8-27B-FP8 \
  --max-model-len 50000 \
  --kv-cache-dtype fp8 \
  --reasoning-parser qwen3 \
  --limit-mm-per-prompt '{"image":0,"video":0}' \
  --gpu-memory-utilization 0.90 \
  --speculative-config '{"method":"mtp","num_speculative_tokens":2}' \
  --port 8000
```

The startup log should contain `speculative_config=SpeculativeConfig(method='mtp', num_spec_tokens=2, ...)`. After the server is healthy:

```bash
curl -s localhost:8000/metrics | grep -c spec_decode_num_draft   # want >= 1
```

If the log says the checkpoint has no MTP weights, the export dropped them. Use an n-gram control instead:

```json
{"method":"ngram","num_speculative_tokens":3,"prompt_lookup_max":4}
```

Run a second instance, or restart, **without** `--speculative-config` for the baseline. `mtp_ab.sh` does both. `K` is `num_speculative_tokens` and defaults to 2:

```bash
chmod +x mtp_ab.sh && rm -f results.tsv
./mtp_ab.sh          # k=2
K=3 ./mtp_ab.sh      # k=3
```

Or point the bench at a server you already started. The third argument is the row tag. At conc=1 the script also writes `out_<tag>_c1.txt` (the first prompt's completion) for a losslessness check.

```bash
python mtp_bench.py http://localhost:8000 1 mtp_k3
python mtp_bench.py http://localhost:8000 8 mtp_k3
python mtp_bench.py http://localhost:8000 32 mtp_k3
diff -q out_base_c1.txt out_mtp_k3_c1.txt && echo IDENTICAL
```

MTP is lossless. Greedy outputs must match token for token. Each run appends one row to `results.tsv`.

If `spec_decode_num_accepted_tokens_total` is missing in your build, `curl localhost:8000/metrics | grep -i spec` and adjust the names in `metrics()`.

## Columns

| Column | Meaning |
| --- | --- |
| `out` | Sum of completion tokens across requests |
| `agg` | `out / wall_time` |
| `decode/req` | Mean of `completion_tokens / time_after_first_token` per request |
| `TTFT` | Mean time to first content token |
| `acc_rate` | Accepted draft tokens / draft tokens over the run. NaN when spec decode counted no drafts |
| `acc_len` | `1 + accepted_tokens / num_drafts`: mean tokens emitted per verify step |

`acc_rate` is over the whole mix. Code and structured prompts accept more drafts than the prime-list and translation prompts. Prompt 8 (repeat a fixed phrase 30 times) is the ceiling check: acceptance there should be ~0.95 or higher. Below that, drafts are not being verified correctly. The bench does not break acceptance out per prompt.

On Blackwell, single-stream decode is expected around 1.5–2.2× versus spec-off, with acceptance about 0.7–0.85. Gain shrinks as concurrency rises and the GPU fills up. Sweep `K` of 1, 2, and 4 to find the knee; k=3 is the measured point below.

A `base` row with a real acceptance rate is not a spec-off baseline. `result.txt` is one such session: the MTP launch logged no `SpeculativeConfig`, and the `base` and `mtp` tags both sat near 111 tok/s/req with acceptance ~0.8. Those rows are in `results.tsv` and are not used below.

## Results

### RTX Pro 6000 — spec off vs MTP k=3

Qwen3.8-27B-FP8, `max_tokens=512`.

| conc | base agg | MTP agg | agg × | base tok/s/req | MTP tok/s/req | per-req × | base TTFT | MTP TTFT | acc_rate | acc_len |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 45.7 | 102.7 | 2.25 | 45.9 | 115.6 | 2.52 | 44 | 555* | 0.820 | 3.46 |
| 8 | 294.9 | 587.6 | 1.99 | 43.1 | 106.7 | 2.48 | 83 | 384* | 0.780 | 3.34 |
| 32 | 990.3 | 1736.9 | 1.75 | 36.2 | 72.3 | 2.00 | 157 | 198 | 0.761 | 3.28 |

\* First requests after launch: cold CUDA-graph / JIT warmup, not spec overhead. A previous warm session showed 56–88 ms TTFT. The conc=1 aggregate (102.7 tok/s) is pulled down by that 555 ms; per-request decode (115.6 tok/s) is the clean number.

Read: about 2.5× single-stream decode, about 2× per request at conc=32, acceptance 0.82 → 0.76 as the batch grows.

Raw MTP k=3 rows:

| conc | out tokens | agg tok/s | decode tok/s/req | TTFT ms | acc_rate | acc_len |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 512 | 102.7 | 115.6 | 555 | 0.820 | 3.46 |
| 8 | 3456 | 587.6 | 106.7 | 384 | 0.780 | 3.34 |
| 32 | 14002 | 1736.9 | 72.3 | 198 | 0.761 | 3.28 |

### RNGD — spec decode off

Qwen3.8-27B-FP8, `max_tokens=512`. These are the last three rows of `results.tsv`. Acceptance is NaN because no draft tokens were counted.

| conc | out tokens | agg tok/s | decode tok/s/req | TTFT ms | acc_rate | acc_len |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 512 | 56.0 | 56.6 | 89 | — | — |
| 8 | 3602 | 373.3 | 54.4 | 191 | — | — |
| 32 | 13931 | 1187.6 | 47.1 | 384 | — | — |
