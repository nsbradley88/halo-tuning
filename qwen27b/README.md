# Qwen3.8-27B on llama.cpp Vulkan (Strix Halo laptop, 70 W)

Model: Qwen3.8-27B (dense, `qwen35`): 48 gated-delta-net (linear attention) layers plus a gated full-attention layer
every 4th layer (head_dim 256, 24 query / 4 KV heads), and an MTP head shipped as a separate GGUF. Every generated token
reads all 27B weights, so plain decode is bandwidth-bound and speculation is the main decode lever.
Engine: llama.cpp `529ac893e` + our [27B patch series](../patches/llama.cpp-q27/), Vulkan (RADV), f16 KV, 64K context.
Presets: [configs/qwen27b/](../configs/qwen27b/).

## Final numbers

These are the llama.cpp numbers. Our HIP engine Talus is now faster on every row; see [below](#talus-our-hip-engine-on-the-same-model).

Chat runs at temperature 1.0, code and edits are greedy. Edits ask the model to rewrite a ~9K-character source file and
return it whole (an agentic edit).

| tok/s | Stock llama.cpp, UD-Q4_K_XL, MTP 4 | **UD-IQ4_XS + patches + MTP 5 + ngram-mod** | Code profile (DFlash2) |
|---|---|---|---|
| Chat | 27.9 | **40.6** | 40.2 |
| Code | ~33 | **46.8–47.4** | **51.6** |
| Agentic edits | ~33 | **76.3–80.0** | 72.7 |
| Decode @32K (MTP) | ~23 | **33–35** | 32.2 |
| Prompt @2K / 32K / 60K | ~290 / 121–202 / ~107 | **~315 / ~260 / ~222** | 318 / 266 / – |
| Plain decode (no speculation) | 12.0 | 14.0 (bandwidth ceiling) | |
| Real agentic sessions (opencode, ~25K mean depth) | ~30 | **33.5**, prompt work −40% with the host prompt cache | 30.8 |

Quality gates on both adopted configs: needle 15/15 (60K prompt, 3 seeds), agentic suite 5/5. UD-IQ4_XS is KLD 0.024 vs
Q8_0 (Q8_0 logits as reference); the integer-dot kernels left it at 0.0246.

Prompt speed at depth depends a lot on the ubatch on stock Vulkan: stock is 135 tok/s at 32K with `-ub 1024` but 202 at the
default 512. Our patched build is flat across 512/1024/2048, and the server default (512) is best.

## What worked

1. **Smaller quant: UD-IQ4_XS** (13.3 GiB, KLD 0.024). Plain decode is at the bandwidth ceiling, so −19% bytes per token
   gave +10% end to end. UD-IQ3_S (11.2 GiB, KLD 0.058) was no faster, because its kernels were slower per byte.
2. **MTP draft vocabulary** (patch 0001/0006, `--spec-draft-vocab`): Qwen's vocabulary is 248K tokens. The MTP head now scores
   only the 64K most frequent ones (built from code, docs, prose and model-written agent text; the file and builder are in
   [configs/](../configs/)). A draft step goes from 4.2 to 2.2 ms with no loss of acceptance; 16K/32K subsets cost
   acceptance. Verification is unchanged, so output stays exact. +10% chat, +9% code.
3. **Vulkan prefill at depth** (patches 0002–0005):
   - **KV-slice prefill ordering.** In the cache layout each workgroup streamed the whole KV range alone and used only a
     quarter of each row. Now the KV head goes first and slices are cache-sized. FA per call at 32K went from 1,026 to 93 ms.
   - **Transposed concat.** The delta-net conv-state concat (the same bug we fixed for GLM) is now two tiled copies,
     573 → 80 ms per ubatch.
   - **Two coopmat1 FA micro-optimizations** by Nathan Wilson.
   - Result: prompt +30% at 32K and +109% at 60K against stock's best ubatch.
4. **Cheaper verify rounds** (patches 0008–0011): the delta-net wrote its rollback snapshots into its own output and a copy
   moved ~30 MB per layer into the cache. Writing them straight into the cache (our GLM patch 0009 idea) took the 5-column
   matvecs back to their isolated speed (IQ4_XS 157 → 100 µs). Integer-dot (q8_1) matvecs for the IQ3_S/IQ3_XXS/IQ2_S/IQ2_XS
   tensors inside UD-IQ4_XS replaced their float path. Verify overhead over a 1-token decode: 20.9 → 10.6 ms.
   +13% chat, +11% code, +13% at 32K.
5. **Flat grouped-query attention for 2–8-token verify** (patch 0012): each verify token re-read the KV cache. Now the
   tokens and heads of a KV head share one pass. 32K attention per verify: 23.9 → 13.0 ms (MTP 5 tokens), 42.9 → 17.7 ms
   (DFlash2 8 tokens). +13% decode at 32K with MTP, +23% with DFlash2.
6. **Matvec up to 16 columns on RDNA3/4** (patches 0013/0015): makes wide verify batches cheap (8-token MTP drafts went
   from 16.9 to 33.4 chat tok/s), though acceptance still makes 4–5 tokens the best draft length.
7. **`ngram-mod` drafting next to MTP** (stock, from the Halogen idea of prompt lookup): edits 46 → 70–80 tok/s at no cost on
   chat/code. It copies whole runs of the file being edited. It hurt on MiMo and GLM, whose drafters already accept
   edits well, so gate it per model.
8. **Host prompt cache** (`--cache-ram 16384`, stock): in agent sessions one server slot was evicted by side requests and whole
   conversations were re-read. With the cache, prompt tokens processed fell 346K → 137K over the agentic suite.

## What didn't

| Change | Result |
|---|---|
| ROCm/HIP instead of Vulkan (same commit) | decode −10%, longer drafts get slower (n4 < n2); HIP prefill is only faster than *stock* Vulkan at depth |
| DFlash2 for chat | 30–40 vs MTP 40.6; DFlash2's trained block is 8, so `--spec-draft-n-max 15` silently means 7 |
| MTP 6–8-token drafts | acceptance falls faster than verify gets cheaper |
| UD-IQ3_S, Q4_0 | no faster than IQ4_XS, worse KLD |
| Draft vocab 16K / 32K | acceptance loss |
| `--spec-draft-p-min` | no gain |

## Why agentic sessions are slower than the mixes

Over real opencode sessions the decode rate averages ~31–33 tok/s while the mixes do 40–80. Sessions sit at a mean depth
of ~25K (each round ~0.9 ms slower per 1K of context), and reasoning text accepts only ~0.5 of drafts (answers ~0.66).
Prompt processing was 31% of server time without the prompt cache, half of it avoidable re-reads.

## Talus (our HIP engine) on the same model

Talus (a fork of Strata with multi-model support; not yet public) now runs this model natively. Same file (UD-IQ4_XS),
same harness, one session; all three run MTP 5 + the 64K draft vocabulary + n-gram drafting:

| | **Talus** | llama.cpp (this page) | llama.cpp ([unified build](../patches/llama.cpp-halo/)) |
|---|---|---|---|
| Chat / code / edits | **43.6 / 51.9 / 86.5** | 40.7 / 47.9 / 80.5 | 38.4 / 47.2 / 82.6 |
| ms per round (chat / code / edits) | **90.9 / 89.5 / 121.9** | 97.1 / 96.6 / 184.0 | 95.6 / 96.5 / 179.8 |
| Prompt @2K / 32K | **349 / 309** | 321 / 267 | 322 / 268 |
| Decode @2K / 32K (128-token answer) | **38.2 / 41.5** | 26.0 / 34.6 | 27.3 / 34.6 |

Talus now wins every row. On chat and code it gets the same tokens per round as llama.cpp (3.96 vs 3.95, 4.64 vs 4.63);
the gain is a 6–7% cheaper round. On edits it verifies shorter n-gram windows (10.5 vs 14.8 tokens per round) at two
thirds of the cost. What made the difference: the i-quant matvecs were bound on emulated byte operations, and a native
byte-permute table lookup cuts IQ4_XS from ~196 to ~72 instructions per 32 weights (bit-identical); a per-arch table of
rows per wave, occupancy and column groups for each quant type and column count makes 5–16-column verify passes cheap,
so a 16-row edit window costs ~2.0× a decode step (was 3.3×); MTP 5 instead of 4; and sampled drafts drawn from the
sampler's min-p support (chat +8%). Quality: greedy output byte-identical to plain greedy, KL vs a CPU reference 0.0025,
needle 15/15, agentic 5/5 (on a re-run; the first run was 4/5, one sampled repetition loop hit the output limit).

The two llama.cpp builds differ on chat through numerics, not speed: their rounds cost the same (the unified build's are
1–2% faster), but its different numerics sample a different chat text that accepts fewer drafts (3.67 vs 3.95 tokens per
round).

## Reproduce

```sh
git clone https://github.com/ggml-org/llama.cpp && cd llama.cpp && git checkout 529ac893e
git am /path/to/halo-tuning/patches/llama.cpp-q27/*.patch
cmake -B build -DGGML_VULKAN=ON -DCMAKE_BUILD_TYPE=Release && cmake --build build -j
# model: unsloth/Qwen3.8-27B-GGUF Qwen3.8-27B-UD-IQ4_XS.gguf and MTP/mtp-Qwen3.8-27B-Q4_0.gguf
llama-server -m Qwen3.8-27B-UD-IQ4_XS.gguf --jinja -ngl 99 -fa on -c 65536 --parallel 1 --cache-ram 16384 \
  --spec-type draft-mtp,ngram-mod -md mtp-Qwen3.8-27B-Q4_0.gguf -ngld 99 --spec-draft-sampling probabilistic \
  --spec-draft-n-max 5 --spec-draft-vocab qwen-dv-65536.bin
```

On stock llama.cpp: UD-IQ4_XS, MTP, `ngram-mod`, probabilistic drafting and the prompt cache all work (chat ~32, code ~37 with
MTP 4 drafts). The draft vocabulary, prefill-at-depth fixes, verify-round fusions, integer-dot IQ matvecs and the flat verify
attention need the patches.
