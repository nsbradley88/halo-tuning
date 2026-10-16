# GLM-5.3-Flash on llama.cpp Vulkan (Strix Halo laptop, 70 W)

Model: zai-org GLM-5.3-Flash (320B total / 18B active MoE; 45 layers + 1 MTP layer: 34 KDA linear-attention layers and 11
DSA sparse-attention layers (MLA, lightning indexer, top-2048 keys), mHC residual streams, 288 experts top-8 plus a shared
expert, 3 dense blocks). Starting quant: Unsloth `UD-IQ3_XXS` (112 GiB).
Engine: llama.cpp `529ac893e` + our [unified patch series](../patches/llama.cpp-halo/) (recommended; the first numbers below
are from `efe3626b1` + the [GLM series](../patches/llama.cpp-glm/)), Vulkan (RADV), f16 KV, 65,536 context.
Presets: [configs/glm/](../configs/glm/) (`glm-5.3-flash-halo.env` for the unified build).

This is the largest model we run: ~110 GiB of weights on a 128 GB machine. It only fits with the GUI off and a 120 GiB GTT pool
(see [Memory](#memory)), and its decode speed is set by how many bytes each token reads.

## Final numbers

| | Release quant, stock llama.cpp, no MTP | **a4e3s + patches + MTP** |
|---|---|---|
| Chat decode (temp 1.0, 6-task mix, ~600 tokens each) | — (no MTP: ~15) | **28.5–29.6** |
| Code decode | — | **31–32** |
| Decode @ 2K / 32K (llama-server, 128 tokens) | 15.4 / 13.3 | 24.3 / **28.2** |
| Prompt @ 2K / 32K | 130 / 98 | **229–233 / 208** |
| 60K-token prompt | 86 (after round 3) | **185–198** |
| Draft acceptance (chat / agentic coding) | — | 0.74 / 0.83 |
| Agentic suite (5 Go tasks, opencode) | — | **10/10** over two runs (17 and 26 min) |
| Needle (15, ~60K prompt) | — | 15/15 |

How we got there, in steps (each A/B'd against a same-session baseline):

| Step | Chat decode | Prompt 2K / 32K |
|---|---|---|
| Release, stock, ub 512, no MTP | ~15 (no MTP) | 130 / 98 |
| a4 requant + MTP (2 tokens) + sparse FA for small batches | 23.3 | — |
| a4e3 (imatrix), ub 768, 64 heads folded as 2 × 32 in FA | 23.6 | 146 / 105 |
| sparse prefill FA, KDA concat fix, a4e3s, ub 1024, draft ubatch 128 | — | 172 / 147 |
| int8 coopmat matmul for IQ2_S / IQ3_S experts | — | 203 / 172 |
| decode indexer + sparse FA kernels, verify matvecs, fusions, probabilistic drafting, draft vocab | **28.5–29.6** | **229 / 208** |

## Unified build (2026-10-16)

The GLM series ported onto the Qwen3.8-27B branch ([unified series](../patches/llama.cpp-halo/)). Same session against the
`efe3626b1` build with the same flags, on the 27B harness (chat: 6 prompts at temperature 1.0; code: 6 greedy; edits: three
~9 KB C++ files rewritten whole, greedy).

| tok/s | `efe3626b1` + GLM series | **Unified build + workgroup-rule env** |
|---|---|---|
| Chat | 27.8–28.0 (0.68) | **28.6**, +2.5% |
| Code | 29.2–29.4 (0.74) | **30.1**, +2.5% |
| Agentic edits | 29.6–29.8 | 30.0 |
| Decode @2K / 32K (128 tokens) | 24.7 / 27.1 | **25.8 / 29.9**, +4% / +10% |
| Prompt @2K / 32K | 236 / 213 | 238 / 216 |

Quality: KLD vs the release logits (2,048 context, `-ub 3` = the verify matvec path) 0.04194 → **0.04191**. Gate on the
adopted config: **needle 15/15** at 60K (prompt 205–207 tok/s), **agentic 5/5** (1,098 s).

- **Set `GGML_VK_DMMV_LARGE_MAX_M=1024 GGML_VK_DMMV_LARGE_MIN_K=8192`.** The GLM series' RDNA3 matvec workgroup rule
  (patch 0018 there) is an opt-in knob on the unified build; without it decode at 2K falls to 22.0.
- The gain is at depth, from the merged FA / matvec work. GLM's experts were already near the bandwidth roofline: the new
  integer-dot IQ2_S / IQ3_S expert matvec gives ~1–3% here (16% in isolation at 5 tokens; GLM verifies 3).
- **Memory is unchanged and still tight:** 7–9 GiB MemAvailable while serving, the same as the older build and far below
  our usual 20 GiB rule. Keep the MemAvailable watchdog ([Memory](#memory)); the host prompt cache was not tested (no headroom).

| Also tried on the unified build | Result |
|---|---|
| MTP 3-token drafts | chat 26.9, code 28.1: still slower than 2 |
| `ngram-mod` next to MTP | edits **−12%** (30.0 → 26.4) |
| Integer-dot expert matvec off | chat 28.2, code 29.1 |

## What worked

1. **Requant the non-expert weights, keep the experts.** Summing bytes per token: GLM reads ~8 GB per generated token, and the
   release kept attention, KDA, the shared expert and the dense FFN at Q6_K. Step by step, every other tensor pinned to its
   current type so it is copied byte for byte:
   - **a4:** attention / KDA / shared-expert / dense FFN Q6_K → Q4_K. +13% decode, −1.6 GiB, KLD 0.037.
   - **a4e3:** the three IQ4_XS `ffn_down_exps` → IQ3_S, using our own imatrix. −0.7 GiB, KLD 0.040.
   - **a4e3s:** 256-aligned Q8_0 → Q6_K, output Q6_K → Q5_K, token_embd → Q4_K. KLD **0.043** (int8 activations
     included), ~109.5 GiB.

   The exact per-tensor types are in [configs/glm/glm-a4e3s-tensor-types.txt](../configs/glm/glm-a4e3s-tensor-types.txt).
   Stock `llama-quantize` refuses to copy IQ experts without an imatrix, and `llama-imatrix` failed on split GGUFs with the
   NextN layer; patches 0002 and 0003 fix both.
2. **MTP speculation, 2-token drafts, probabilistic drafting.** GLM's own MTP layer (`--spec-type draft-mtp`) with
   `--spec-draft-sampling probabilistic` (stock; rejection sampling, exact in distribution) gave +5% over greedy drafts. 3-token
   drafts are slower: a longer verify costs more than the extra accepted tokens. Two patches make MTP cheaper:
   - `--spec-draft-ubatch 128` (patch 0006; upstream now has `--spec-draft-n-ubatch`, #30459) shrinks the draft context's
     compute buffer;
   - `--spec-draft-vocab` (patch 0017): the draft head scores only the 64K most frequent tokens instead of 154,880. The
     verify pass is unchanged, so output is exact; 0.9% of chat tokens fall outside the subset and lose their draft. The
     subset is [configs/glm/glm-dv-65536.bin](../configs/glm/), built with
     [make_vocab.py](../configs/glm/make_vocab.py) from model-written agent text, docs, code and Python.
     +10–14% decode at 32K.
3. **Sparse attention that stays sparse on Vulkan.** GLM's DSA layers attend to the 2,048 keys the indexer picks. MTP verify
   batches (2–3 tokens) fell off Vulkan's sparse path onto dense FA with a mask: decode at 32K
   collapsed to 13 tok/s with MTP on. Fixes, in order:
   - per-row sparse FA for small batches (patch 0001), 32K 13 → 21 tok/s;
   - 64 heads folded as 2 × 32 in FA, 6.9 → 2.8 ms per token at 32K;
   - sparse FA for prefill batches (0004);
   - a dedicated sparse FA kernel and a decode lightning-indexer kernel (0014, 0015): +7% decode and +10% prompt at 32K,
     60K prompt 161 → 185–198 tok/s.
4. **int8 coopmat matmul for the IQ2_S / IQ3_S experts** (patch 0007, dense and mul_mat_id). Prefill is 58% routed-expert
   matmuls, and Vulkan had int8 coopmat only for Q4_0–Q6_K / IQ4. The key was one thread per 32-value sub-block: four
   threads × 8 values was load-bound on the scattered IQ2_S codebook loads. Expert matmuls reached HIP's MMQ speed, and
   prompt went 172 → 203 tok/s at 2K.
5. **Multi-column matvecs for the verify batch** (0016, 0018, 0020): a 3-token MTP verify reads each weight row once for
   all three columns, with an RDNA3 workgroup rule for few-row / long-row matrices.
6. **Fusions** (0008–0013, 0021, 0022): a fused mHC pre-mix op (−990 dispatches per token), KDA state read and written in
   place, KDA prologue inside the gated-delta-net op, RMS_NORM+SCALE and SCALE→SIGMOID→SCALE chains, split-K `hc_fn`. Together
   ~+7–10% decode.
7. **A KDA conv fix** (0005): the conv-state concat read a transposed view, 23 ms per call at 1.8K tokens. Making it
   contiguous first removed that copy cost.
8. **`kho=off` on the Ubuntu 7.0 kernel** (host, see [platform/](../platform/)): kexec-handover reserved ~8.7 GiB as CMA,
   which GPU memory can't use. That was the hidden cause of the OOMs at ~116 GiB.

After all this, the decode roofline is routed experts at 83% of the ~210 GB/s peak and the large dense matvecs at 90–95%:
about 2–3 ms of a 78 ms verify step is left to win exactly. With MTP the chat ceiling is ~38–41 tok/s at 100% bandwidth,
because a 3-token verify reads ~20 distinct experts (each token picks its own 8). 50 tok/s needs a lossy change or another model.

## What didn't

| Change | Result |
|---|---|
| Fewer experts (top-7 / top-6) | KLD 0.059 / 0.100 for ~+5% / ~+10%; too lossy |
| Q3_K attention / KDA (± shared + dense) | KLD 0.090 / 0.102; too lossy |
| ROCm (HIP) instead of Vulkan, same commit, model, flags and session | chat **21.5 vs 29.6**, prompt 32K **117 vs 208**, decode 32K 18.3 vs 28.2; single-token IQ expert matvec 1.46× slower on HIP |
| Porting decode to Talus (HIP) | no gain expected; Talus also lacks KDA / indexer / mHC |
| MTP 3-token drafts | slower than 2 |
| n-gram + MTP drafts | +24% on copy-heavy edits, slower elsewhere |
| q8_0 KV | −7% decode |
| ub 2048 (no MTP) | prompt 213 / 170 but decode −25%; with MTP it does not fit |
| Smaller mul_mat_id tiles, MMVQ / submit / graph-optimizer knobs, draft p-min | no gain |

ROCm note: the community report of ROCm decoding ~1.7× faster on this model (Unsloth MIX build) compared against a Vulkan
build whose GLM path was untuned (pp512 ≈ 60 tok/s). With tuned Vulkan kernels, Vulkan wins both decode and prefill here.

## Memory

- 128 GB box, ~110 GiB model: boot the multi-user target (no GUI) with a **120 GiB GTT pool**
  (`ttm.pages_limit=31457280 ttm.page_pool_size=31457280`) and `kho=off`. With the full serving config (64K, MTP,
  ub 1024) ~8–9 GiB stays free. That is below our usual 20 GiB rule, so a MemAvailable watchdog stops the server below 4 GiB.
- `--load-mode none` (keeps the weights out of the page cache), `--parallel 1`, `-ctxcp 4` (each KDA context checkpoint
  is 146 MiB, default 32), `--cache-ram 0` (prompt cache defaults to 8 GiB of host RAM).
- llama.cpp's fit check reported ~123 GiB free. Before `kho=off`, loads projected at ≥117 GiB failed in TTM allocation; one
  hit the global OOM killer and one hard-crashed the box. Check `-lv 4`'s `projected to use` line before trying a bigger config.
- The compute buffer grows with context × ubatch (ub 1024: ~1.3 GiB at 32K, ~2 GiB at 64K). Most of that is ggml-alloc
  fragmentation (~256 MiB live), which is what blocks ub 2048 with MTP.

## Box lessons from this model

- **Never compile on the box while benchmarking.** CPU and GPU share the 70 W package budget, and a build halves GPU clocks.
- **Sustained power drifts 69 → ~57 W** as the chassis warms through the day (skin-temperature limit). Long A/Bs must
  alternate runs; better cooling would be worth ~5–10%.
- **Don't edit a bash script while it runs.** bash reads it incrementally.

## Reproduce

```sh
git clone https://github.com/ggml-org/llama.cpp && cd llama.cpp && git checkout 529ac893e
git am /path/to/halo-tuning/patches/llama.cpp-halo/*.patch
cmake -B build -DGGML_VULKAN=ON -DCMAKE_BUILD_TYPE=Release && cmake --build build -j

export GGML_VK_DMMV_LARGE_MAX_M=1024 GGML_VK_DMMV_LARGE_MIN_K=8192
llama-server -m GLM-5.3-Flash-a4e3s-00001-of-00004.gguf --jinja -ngl 99 -ub 1024 --load-mode none -c 65536 \
  --parallel 1 -ctxcp 4 --cache-ram 0 --spec-type draft-mtp --spec-draft-n-max 2 --spec-draft-ubatch 128 \
  --spec-draft-sampling probabilistic --spec-draft-vocab glm-dv-65536.bin
# client: temperature 1.0, top_p 0.95; reasoning_effort via chat_template_kwargs
```

The older build (`efe3626b1` + the [GLM series](../patches/llama.cpp-glm/)) runs the same command without the environment.

Requant (from the release, with an imatrix; ours: 150 × 512-token chunks of mixed docs / code / agent text):

```sh
llama-quantize --allow-requantize --keep-split --imatrix glm.imatrix \
  --tensor-type-file glm-a4e3s-tensor-types.txt --output-tensor-type q5_k --token-embedding-type q4_k \
  GLM-5.3-Flash-UD-IQ3_XXS-00001-of-00004.gguf GLM-5.3-Flash-a4e3s.gguf Q4_K_M
```

We built a4e3s in three passes (attention without an imatrix, then the experts, then the Q8_0 / output tensors). One pass
with the final type file gives the same types; its KLD may differ slightly from our measured 0.043.

On stock llama.cpp (≥ `efe3626b1`): MTP with 2-token probabilistic drafts and the f16 64K configuration work, but expect
roughly the first rows of the step table. The sparse FA paths, the IQ2_S / IQ3_S int8 matmul, the decode indexer, the
verify matvecs, the fusions and the draft vocabulary need the patches. The requant needs patch 0002 (or an imatrix for every tensor).
