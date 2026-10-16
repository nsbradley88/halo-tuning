# MiMo-V2.6-Flash on llama.cpp Vulkan (Strix Halo laptop, 70 W)

Model: Xiaomi MiMo-V2.6-Flash-MOPD (309B total / 15B active MoE; 48 layers + 3 MTP layers: 9 full-attention layers with K 192 / V 128 and 4 KV
heads, the rest sliding-window-128 with sinks; 256 experts, top-8). Starting quant: the BPW2.5 release (97 GB).
Engine: llama.cpp `529ac893e` + our [unified patch series](../patches/llama.cpp-halo/) (recommended; the first numbers below
are from `a8c9a4e7c` + the [Laguna/MiMo series](../patches/llama.cpp/)), Vulkan, `-fa on`, q8_0 K and V, 196,608 context.
Presets: [configs/mimo/](../configs/mimo/) (`mimo-v2.6-flash-halo.env` for the unified build).

## Final numbers

| tok/s | Release quant, no speculation | **a4 quant + DFlash** |
|---|---|---|
| Code @ ~1K / 31K | 24.9 / 23.4 | **45–47 / 41–43** |
| Prose @ ~1K / 31K | 25.0 / 23.5 | 27.5–28.5 / 24.8–26 |
| Decode (llama-bench tg64) @ 0 / 32K | 26.2 / 24.4 | **30.7–32.0 / 28.4–28.6** (no speculation) |
| Agentic suite (5 Go tasks) | 5/5, 23.9 tok/s | 5/5, 30.7 tok/s; **thinking off: 5/5 in 923 s, 34.2 tok/s** |
| Needle (15) | 15/15 | 15/15 (every adopted config) |

## Unified build (2026-10-16)

The Qwen3.8-27B kernels plus two new patches, on the [unified series](../patches/llama.cpp-halo/). Same session against the
`a8c9a4e7c` build with the same config, on the 27B harness (chat: 6 prompts at temperature 1.0; code: 6 greedy; edits: three
~9 KB C++ files rewritten whole, greedy). Draft acceptance in brackets.

| tok/s | `a8c9a4e7c` build, DFlash n4 | **Unified build + DFlash draft vocab, n4** |
|---|---|---|
| Chat | 28.7 (0.51) | **31.0** (0.54), +8% |
| Code | 35.3 (0.73) | **39.7** (0.70), +13% |
| Agentic edits | 43.8 | **50.6**, +16% |
| Decode @2K / 32K (128 tokens) | 28.3 / 25.1 | **31.8 / 27.7**, +12% / +10% |
| Prompt @2K / 32K | 273–307 / 251–290 | 311 / 284 (unchanged within noise) |

Quality: KLD vs the release logits (2,048 context, `-ub 5` so the expert matvec path is measured) 0.02785 → **0.02784**.
Gate on the adopted config: **needle 15/15** at 81K, **agentic 5/5** (thinking on, 1,412 s). With `--cache-ram 8192`
(also gated) MemAvailable fell to 11 GiB.

What made the difference:

- **Integer-dot expert matvecs** (patches 0044, 0046). Decode and verify (up to 8 tokens) run the experts through
  MUL_MAT_ID matvecs, and Vulkan had integer-dot (q8_1) pipelines there only up to IQ4_XS / IQ1 / K-quants, so MiMo's
  IQ2_XXS / IQ3_XXS / IQ2_XS experts ran on the float path. New pipelines with 4 rows per workgroup: −12 to −27% per
  5-token expert matvec.
- **Draft vocabulary for the DFlash drafter** (patch 0045, `--spec-draft-vocab`). MiMo's drafter has its own 152,576-row
  Q8_0 head (664 MB). Scoring the 64K most frequent tokens covers 99.6–99.9% of held-out text:
  +1–4% on top of the kernels. The subset is [configs/mimo/mimo-dv-65536.bin](../configs/mimo/), built with
  [make_vocab.py](../configs/mimo/make_vocab.py) (same corpus as the 27B subset, tokenized with MiMo's tokenizer).

| Also tried on the unified build | Result |
|---|---|
| DFlash n3 / n5 / n6 | chat 32.1 / 30.2 / 29.1, code 38.2 / 37.9 / 37.3, edits 46.4 / 52.1 / 53.2: n3 for chat-heavy use, n5–6 for edit-heavy use |
| `ngram-mod` next to DFlash | edits **−18%** (50.6 → 41.7): DFlash already accepts 98% on edits |
| KV slice 2048 (the 27B default) / head-outer order | prefill within noise of slice 8192 |

Prefill is unchanged: the expert matmuls dominate it (see [the prefill profile](#prefill-profile-where-the-remaining-time-goes)).

## What worked

1. **Requant the attention, not the experts** (needs our one-line quantize patch, or the uploader's imatrix). Summing the
   bytes per generated token showed attention (Q6_K, all layers) at **47%** of MiMo's 7.7 GiB per token; the experts are
   already ~2.5 bits. Attention Q6_K → **Q4_K** and output head Q8_0 → Q6_K ("a4", 89 GiB): **+17% decode**, KLD 0.027
   vs the release, perplexity +0.4%, agentic 5/5, needle 15/15. Q5_K ("a5"): +9.5%, KLD 0.014.
   How: `llama-quantize --allow-requantize --keep-split --tensor-type-file mimo-a4-tensor-types.txt --output-tensor-type
   q6_k --token-embedding-type q8_0 <release> <out> Q4_K_M` — the override file ([configs/mimo/](../configs/mimo/)) pins
   every other tensor to its current type so it is copied byte-for-byte. Stock llama-quantize refuses to copy the IQ
   experts without an imatrix; patch 0010 makes the check skip tensors that do not change type.
2. **DFlash speculation, 4-token drafts, adaptive pause 1.2** (DFlash: stock; pause: patch). On a4 DFlash beats MTP
   everywhere (agentic 30.7 vs 27.6 tok/s). MiMo's drafter already has a 1024-token window, so it needs no `DFLASH_SWA`.
3. **Thinking off for agentic work** (stock, `--reasoning off`, opt-in preset): every task in 139–300 s (thinking on:
   896–1,945 s), 10.8K tokens instead of 28–36K, 80% draft acceptance; needle 15/15.
4. **KV-sliced FA prefill, slice 8192** (patch): +31% prefill per chunk at 128K, +6% end to end on a 124K prompt; no effect
   below 16K.

## What didn't

| Change | Result |
|---|---|
| Block-sparse decode at Laguna's budget (64 blocks) | **breaks recall** (needle 0/5); 256 blocks matches dense but is slower than dense at 81K |
| Fewer experts (top-7 / top-6) | KLD 0.022 / 0.051 for ~4% / ~9% |
| More non-expert requant (attention IQ4_XS, dense FFN Q4_K, output Q5_K) | +3.5–4.3% decode, but perplexity +0.9% and a weaker agentic run |
| Smaller drafter (Q6_K / Q4_K) | no faster; Q4_K −7% at 31K (acceptance drops) |
| Draft length 3 / 5 / 6, drafter window 512, adaptive threshold 1.6 | no gain over n4 / 1.2 |
| Thinking budget 2048 tokens | task 1 timed out (worse than both on and off) |
| ub 4096 | +6–9% prefill, but with the drafter at 196K only ~13 GiB stayed free and the GPU memory manager crashed (device lost + TTM list corruption; reboot needed) |

## Prefill profile (where the remaining time goes)

pp2048 @ depth 0: expert matmuls (MUL_MAT_ID) **71%**, dense matmuls 18%, FA 3.5%. The expert matmuls run at ~8 TFLOPS vs
~13–20 for dense matmuls: ~20% is mul_mat_id path overhead (it remains with 2,048 rows per expert) and ~25% is tile
underfill at 64 rows per expert. Not an IQ dequant cost (q2_K / q4_K / q8_0 show the same). Fixing it needs a grouped-GEMM
rewrite of the Vulkan MoE path; we have not done it.

## Reproduce

```sh
git clone https://github.com/ggml-org/llama.cpp && cd llama.cpp && git checkout 529ac893e
git am /path/to/halo-tuning/patches/llama.cpp-halo/*.patch
cmake -B build -DGGML_VULKAN=ON -DCMAKE_BUILD_TYPE=Release && cmake --build build -j

export GGML_VK_FA_WAVE32=1 GGML_VK_FA_KV_SLICE=8192 LLAMA_SPEC_ADAPTIVE=1.2
llama-server -m MiMo-V2.6-Flash-MOPD-a4-00001-of-00003.gguf -ngl 999 -fa on -fit off --parallel 1 --ctx-size 196608 \
  -ctk q8_0 -ctv q8_0 --ubatch-size 2048 --batch-size 2048 --cache-ram 0 --jinja --temp 1.0 --top-p 1.0 --top-k 20 \
  --min-p 0.0 -md dflash-MiMo-V2.6-Flash-MOPD-Q8_0.gguf -ngld 999 --spec-type draft-dflash --spec-draft-n-max 4 \
  --spec-draft-vocab mimo-dv-65536.bin
# agentic preset: add --reasoning off. Keep ub 2048 here (memory). No ngram-mod.
```

The older `a8c9a4e7c` + [Laguna/MiMo series](../patches/llama.cpp/) build runs the same command without
`--spec-draft-vocab`. On stock llama.cpp: the a4 quant (if you can requantize), DFlash, thinking off and dense attention all
work; the adaptive pause, KV slicing, integer-dot expert matvecs and the draft vocabulary need the patches.
