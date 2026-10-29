#!/usr/bin/env python3
"""Benchmark a running vLLM server, with or without MTP speculative decoding.

Usage:
    python mtp_bench.py BASE_URL CONCURRENCY TAG

Example:
    python mtp_bench.py http://localhost:8000 8 mtp_k3

What it does
    1. Read the served model id from GET /v1/models.
    2. Take CONCURRENCY prompts from a fixed set of eight seeds (the list
       repeats once CONCURRENCY is greater than 8).
    3. Send them together as greedy chat completions (temperature 0, seed 0,
       thinking off) and time each one.
    4. Subtract vLLM's /metrics counters from before the run to after it.
    5. Print one tab-separated row and append that same row to results.tsv.
       At CONCURRENCY 1, also write the first completion to out_<TAG>_c1.txt
       so a baseline and an MTP run can be diffed.

max_tokens=512 caps generated tokens only. The seeds are short (about 20-40
tokens) and are not limited by that setting. out=512 at concurrency 1 means
the reply hit the cap; a smaller total at higher concurrency means some
replies stopped at end-of-sequence first.

Row columns
    out          sum of completion tokens
    agg          out / wall-clock seconds
    decode/req   mean per request of completion_tokens / time after first token
    TTFT         mean time to first content token, in milliseconds
    acc_rate     accepted draft tokens / draft tokens (nan if spec decode is off)
    acc_len      1 + accepted_tokens / num_drafts
                 mean tokens emitted per verify step (nan if spec decode is off)
"""
import sys, time, json, re, requests, concurrent.futures as cf
BASE, CONC, TAG = sys.argv[1].rstrip("/"), int(sys.argv[2]), sys.argv[3]
MODEL = requests.get(f"{BASE}/v1/models").json()["data"][0]["id"]
MAX_TOK = 512

SEED = [
 "Write a Python function that parses an ISO-8601 timestamp without the datetime module, with pytest tests.",
 "Explain NCCL ring all-reduce and give its bandwidth/latency cost model.",
 "List the first 40 primes and the sum of each consecutive pair.",
 "Translate to French, back to English, then summarize the drift: 'The quick brown fox jumps over the lazy dog near the riverbank at dawn.'",
 "Write a SystemVerilog Q16.16 fixed-point multiplier with saturation and a short testbench.",
 "Describe vLLM paged-attention KV layout and block-table mapping.",
 "Produce a JSON schema for a GPU-rental REST API plus one example payload.",
 "Repeat 'speculative decoding accepts drafts' 30 times, numbered.",
]
PROMPTS = (SEED * ((CONC + 7) // 8))[:CONC]

def metrics():
    txt = requests.get(f"{BASE}/metrics").text
    g = lambda n: sum(float(v) for v in re.findall(rf'^vllm:{n}(?:\{{[^}}]*\}})?\s+([0-9.e+]+)', txt, re.M))
    return dict(draft=g("spec_decode_num_draft_tokens_total"),
                acc=g("spec_decode_num_accepted_tokens_total"),
                drafts=g("spec_decode_num_drafts_total"))

def one(prompt):
    t0 = time.perf_counter(); first = None; text = []; usage = None
    r = requests.post(f"{BASE}/v1/chat/completions", stream=True, json={
        "model": MODEL, "messages": [{"role": "user", "content": prompt}],
        "max_tokens": MAX_TOK, "temperature": 0, "seed": 0, "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"enable_thinking": False}})
    for line in r.iter_lines():
        if not line.startswith(b"data:"): continue
        b = line[5:].strip()
        if b == b"[DONE]": break
        d = json.loads(b)
        if d.get("usage"): usage = d["usage"]
        ch = d.get("choices")
        if ch and ch[0]["delta"].get("content"):
            if first is None: first = time.perf_counter()
            text.append(ch[0]["delta"]["content"])
    t1 = time.perf_counter()
    n = usage["completion_tokens"] if usage else len(text)
    f = first or t1
    return n, f - t0, t1 - f, "".join(text)

m0 = metrics(); t0 = time.perf_counter()
with cf.ThreadPoolExecutor(CONC) as ex: res = list(ex.map(one, PROMPTS))
wall = time.perf_counter() - t0; m1 = metrics()

toks = sum(r[0] for r in res)
ttft = 1000 * sum(r[1] for r in res) / len(res)
dec  = sum(r[0] / max(r[2], 1e-9) for r in res) / len(res)
dd, da, dn = (m1[k] - m0[k] for k in ("draft", "acc", "drafts"))
acc_rate = da / dd if dd else float("nan")
acc_len  = 1 + da / dn if dn else float("nan")     # mean tokens emitted per verify step                                                                   

line = f"{TAG}\tconc={CONC}\tout={toks}\tagg={toks/wall:.1f}tok/s\tdecode/req={dec:.1f}tok/s\tTTFT={ttft:.0f}ms\tacc_rate={acc_rate:.3f}\tacc_len={acc_len\
:.2f}"
print(line); open("results.tsv", "a").write(line + "\n")
if CONC == 1:
    open(f"out_{TAG}_c1.txt", "w").write(res[0][3])
