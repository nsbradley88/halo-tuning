# Laguna-S-2.1 on llama.cpp Vulkan (Strix Halo laptop, 70 W)

Model: poolside Laguna-S-2.1 (118B total / ~8B active MoE; 48 layers: 12 full-attention + 36 sliding-window-512; 256
experts, top-10). Quant: a "lean" requant of Unsloth's UD-Q5_K_XL (67 GiB; KLD 0.058 vs UD-Q4_K_XL logits).
Engine: llama.cpp with our [patch series](../patches/llama.cpp/), Vulkan, RADV (Mesa), `-fa on`, q8_0 K and V,
262,144 context. Presets: [configs/laguna/](../configs/laguna/).

## Final numbers

| | Before | After |
|---|---|---|
| Decode @ 128K | 19.5 tok/s | **26.4** (+35%) |
| Decode @ 245K | 15.0 tok/s | **24.2** (+61%) |
| Prefill @ 128K (2048-token chunks) | 155 tok/s | **191** (+23%) |
| Code generation with DFlash @ ~1K / 31K / 124K / 235K | — (DFlash unusable past ~8K) | **40.6 / 37.5 / 34.0 / 34.9** |
| Prose with DFlash (adaptive pause) | — | 30.9 / 27.9 / 25.7 (≈ no-speculation speed) |
| Agentic suite (5 Go tasks) | 5/5, 27.6 tok/s | 5/5, 29.2 tok/s; **727 s with thinking off** (1,632 s on) |
| Short-context decode | 33.8 | 33.4–34.2 (bandwidth limit) |

Quality: needle 15/15 greedy with every adopted config (14/15 in a later same-session control; misses are a single wrong
digit at 10% depth).

## What worked

1. **Block-sparse decode attention (Quest-style)** — *patch*. The 12 full-attention layers are the whole long-context cost.
   Per block of 64 keys we keep the element-wise max/min of the keys, bound q·k per block, and attend only to the top 64
   blocks (4,096 tokens) plus the first 128 and last 1,024 tokens. Full-attention time at 128K: 18.8 → 3.7 ms/token.
   `LLAMA_SPARSE_ATTN=64,64,16384` (blocks, block size, minimum context).
2. **KV-sliced flash-attention prefill** — *patch*. Split FA's KV range into 4,096-token slices processed in order so
   concurrently running workgroups share a cache-sized slice: 2× faster FA at 245K, +23% end-to-end prefill at 128K.
   `GGML_VK_FA_KV_SLICE=4096`.
3. **A sliding window for the DFlash drafter** — *patch*. The Laguna DFlash drafter attends to the whole context, and its
   predictions fall apart as context grows (0.2% acceptance at 31K — the "DFlash collapses past 8K" people report).
   Limiting it to the last 1,024 tokens: ~70% acceptance at 31K and good acceptance at 124K. Window sweep at 31K (code
   tok/s): 512 → 40.0, 1024 → 39.4, 2048 → 38.9, 4096 → 32.4, 8192 → 16.7, full → 10.4. `DFLASH_SWA=1024`.
4. **Adaptive speculation pause** — *patch*: pause drafting when the moving average falls below 1.2 accepted tokens per
   round, so prose stays at no-speculation speed. `LLAMA_SPEC_ADAPTIVE=1.2`. Draft length 4 is best; 8 is a cliff.
5. **ub 4096** (stock): +7–11% prefill at short context, +3–6% at 32K, identical at 235K, no GPU timeout; 27+ GiB free.
6. **Thinking off for agentic work** (stock, `--reasoning off`, opt-in preset): agentic 5/5 in 727 s vs 1,632 s; needle
   13/15 vs 14/15 with thinking on (the template's forced empty think block costs ~1 needle in 15).
7. **amdgpu `lockup_timeout=10000,60000,10000,10000`** (host): the default 2 s compute timeout kills long prefill chunks
   at 200K+ context ("device lost").

## What didn't

- **Weights are at the bandwidth limit.** Every big matmul in decode already runs at ~210 GB/s. Requants (3-bit experts,
  4-bit attention) and fewer routed experts (8/7/6 of 10) bought 3–21% speed for 1.4–3.7× the KLD.
- **n-gram speculation** hurts real agentic sessions (23% acceptance).
- **The HF config's RoPE scaling (YaRN factor 128) vs the GGUF's (32)**: +6.8% worse perplexity at 64K; keep 32.
- **Serving 4 requests at once** (`--parallel 4 -kvu`): +3% total throughput, first answer at 86 s instead of 28 s.

## Two findings worth knowing beyond Laguna

- **Test speculative decoding greedily.** At temperature 1 we saw rare needle misreads only with speculation on and hunted
  a bug; single-token vs batched decode differ by KLD ~0.015, but both score the same against reference logits, and at
  temperature 0 every config scored 15/15. It was sampling variation.
- **On the Vulkan backend, extra per-layer side-chain ops inside the forward graph corrupted the forward pass at depth**
  (a single-token NaN past ~30K) even though every op passed `test-backend-ops`. Doing that bookkeeping on the host after
  each ubatch (cheap on unified memory) fixed it.

## Reproduce

Apply [the patch series](../patches/llama.cpp/) to llama.cpp `a8c9a4e7c`, build with `-DGGML_VULKAN=ON`, then:

```sh
export GGML_VK_FA_WAVE32=1 GGML_VK_FA_KV_SLICE=4096 LLAMA_SPARSE_ATTN=64,64,16384 DFLASH_SWA=1024 LLAMA_SPEC_ADAPTIVE=1.2
llama-server -m Laguna-S-2.1-Q4-lean-00001-of-00003.gguf -ngl 999 -fa on -fit off --parallel 1 --ctx-size 262144 \
  -ctk q8_0 -ctv q8_0 --ubatch-size 4096 --batch-size 4096 --cache-ram 0 --jinja --temp 1.0 --top-p 1.0 --top-k 20 \
  --min-p 0.0 -md laguna-s-2.1-dflash-Q8_0.gguf -ngld 999 --spec-type draft-dflash --spec-draft-n-max 4
# agentic preset: add --reasoning off
```

On stock llama.cpp only items 5–7 apply; DFlash itself works but collapses past ~8K without the drafter window.
