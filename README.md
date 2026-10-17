# Halo Tuning

Tuning five large models for local inference on an **AMD Strix Halo** (Ryzen AI Max+ 395, Radeon 8060S / gfx1151,
128 GB unified memory): **Qwen3.8-Flash-Next** on [Strata](https://github.com/Niko1221/Strata), and **Qwen3.8-27B** (dense), **Laguna-S-2.1**,
**MiMo-V2.6-Flash** and **GLM-5.3-Flash** on llama.cpp (Vulkan). Every number here was measured on one machine, and every adopted change had to
pass the same quality gates (a long-context needle test and a 5-task agentic coding suite).

| Model | Engine | Output tok/s (short → long context) | Prompt tok/s | Quality |
|---|---|---|---|---|
| [Qwen3.8-Flash-Next](qwen-strata/) (125B-A6B, UD-IQ4_XS) | Strata v0.1.41 | **58.5–59.5** @8K, 57–58.6 @32K, 40–44.6 @128K | ~1,000–1,067 | agentic 5/5, needle 15/15 |
| [Laguna-S-2.1](laguna/) (118B-A8B, Q4 "lean" requant) | llama.cpp Vulkan + [unified patches](patches/llama.cpp-halo/) | 33–34 plain; **code with DFlash 40.6 / 37.5 / 34.0 / 34.9** @1K / 31K / 124K / 235K; 26.4 @128K, 24.2 @245K plain; unified build: chat 36.6, code 45.2, **edits 79.1** (`ngram-mod`) | 611–653 @0, 465 @31K | agentic 5/5, needle 14–15/15 |
| [MiMo-V2.6-Flash](mimo/) (309B-A15B, BPW2.5 + Q4_K attention) | llama.cpp Vulkan + [unified patches](patches/llama.cpp-halo/) | 30.7–32 plain; **code with DFlash 45–47 / 41–43** @1K / 31K; unified build: chat 31.0, code 39.7, edits 50.6 | 323–332 @0, ~240 @32K | agentic 5/5, needle 15/15 |
| [Qwen3.8-27B](qwen27b/) (27B dense, UD-IQ4_XS) | llama.cpp Vulkan + [27B patches](patches/llama.cpp-q27/) (= first 15 of the unified series); Talus (our HIP engine, not yet public) | llama.cpp: **chat 40.6, code 46.8–47.9 (51.6 DFlash2), agentic edits 76–80**, ~34 @32K (stock: 27.9 / ~33 / ~33). [Talus](qwen27b/#talus-our-hip-engine-on-the-same-model): **chat 43.5, code 55.1, edits 88.5**, 43.2 @32K (DFlash2 code profile: chat 45.6, **code 60.8**) | llama.cpp ~315 @2K, ~260 @32K, ~222 @60K; Talus 349 @2K, 309 @32K | agentic 5/5, needle 15/15 (both) |
| [GLM-5.3-Flash](glm/) (320B-A18B, ~110 GiB requant, KLD 0.043) | llama.cpp Vulkan + [unified patches](patches/llama.cpp-halo/) | **chat with MTP 28.5–29.6**, code 31–32, ~28 @32K (stock: ~15 without MTP); unified build: **29.9 @32K** | 229–233 @2K, 208 @32K, 185–198 on a 60K prompt | agentic 10/10, needle 15/15 |

## The machine — and why it is not a "max settings" Strix Halo

This is an **HP ZBook Ultra G1a 14" laptop**, not a desktop Strix Halo. Under sustained load the package is held at a hard
**70 W** (measured: median 66–69 W, never above 70; GPU clock ~2,070 MHz of its 2,900 MHz maximum). That matters in one place:

- **Prompt processing is compute-bound**, so it runs about **20–25% below** desktop Strix Halo boxes with a 120 W+ power
  limit (Strata's maintainers report ~1,300 tok/s prompt on Qwen; we reach ~1,000–1,067 with the same release and settings).
- **Token generation is memory-bandwidth-bound** (~210 GB/s on every Strix Halo), so it is not power-limited: our output
  speeds match or beat the published desktop numbers.

Setting the GPU performance level to `high` does not help, and the CPU already peaks at 85–95 °C at 70 W in this chassis,
so we did not try to raise the limit. If you have a desktop Strix Halo, expect our generation numbers and better prompt numbers.

Host settings that did help (all models): a **112 GiB GPU memory pool** (`ttm.pages_limit=29360128`), **`amd_iommu=off`**
(+8–9% prompt, +2–8% output on Strata, +7% prompt on llama.cpp; note it drops DMA protection for USB4/Thunderbolt devices),
and a longer **amdgpu compute timeout** (`amdgpu.lockup_timeout=10000,60000,10000,10000`; the default 2 s kills long
prompt chunks at 200K+ context). Details: [platform/](platform/).

## How we tuned

1. **Find the bottleneck before turning knobs.** For decode, sum the bytes each generated token reads per tensor group and
   compare with ~210 GB/s: if the prediction matches the measurement, only fewer bytes (a smaller quant) or speculation can
   help. For prefill, profile per kernel (GGML_VK_PERF_LOGGER on llama.cpp).
2. **Change one thing at a time, interleaved.** Run-to-run noise on this box is ±1 tok/s on output and up to ±15% on short
   prompts, so every comparison repeats the baseline in the same session (and after a reboot, against a same-session
   pre-reboot run: a reboot alone can move numbers ~5%).
3. **Gate everything on quality**, not just KLD/perplexity:
   - *Needle*: five codenames hidden at 10–90% depth of an ~81K-token code prompt, 3 seeds (15 needles).
   - *Agentic suite*: 5 real Go tasks in the charmbracelet/bubbles repo, run through an agent harness (opencode), graded by
     hidden tests.
   - KLD vs a reference quant's logits for any requant.
4. **Check memory headroom with the full serving config** (drafter + full context loaded) and keep ≥ ~20 GiB free. Twice we
   lost the GPU or the whole session to memory pressure (see [platform/](platform/)).

## What generalised across models

| Finding | Qwen | Laguna | MiMo | GLM |
|---|---|---|---|---|
| Speculative decoding is the biggest single decode lever | MTP, `--spec 5 --mtp-window 8192` (+3–9%) | DFlash with a 1024-token drafter window (the "DFlash collapses past 8K" fix) | DFlash, 4-token drafts | MTP, 2-token probabilistic drafts + 64K draft vocab (needed sparse-FA fixes for verify batches) |
| A requant pays only where non-expert weights dominate the bytes per token | — | no (experts dominate) | **yes**: attention Q6_K → Q4_K, +17% decode | **yes**: attention / KDA / shared / dense Q6_K → Q4_K, +13% decode |
| Thinking off for agentic work | **no** (one task looped, 4/5) | yes, opt-in (727 s vs 1,632 s; needle 13/15 vs 14/15) | **yes** (923 s vs ~1,900 s typical; needle 15/15) | — (not tested) |
| Serving several requests at once (batch slots) | no aggregate gain; first answers much later | no (+3%) | — | — (no memory for it) |
| Block-sparse long-context attention | — | **yes** (+35% decode @128K, +61% @245K) | **no** (breaks recall at the same budget) | built in (DSA); needed dedicated Vulkan kernels |
| Fewer routed experts | — | no (KLD 1.4–3.7×) | no (KLD 0.022–0.051) | no (KLD 0.059–0.100) |
| Draft vocabulary (drafter scores the 64K most frequent tokens) | — | no (code −9%; 100K vocab) | yes (+1–4%; 152K drafter head) | **yes** (+10–14% @32K) |
| `ngram-mod` next to the model drafter | — | edits **+103%**, but weak in agentic turns: an edit profile | no (edits −18%) | no (edits −12%) |

The last rows are a warning: a technique that is free on one model can quietly break another. Gate every model separately.

## The 27B methods on the other models (2026-10-16)

We ported the Qwen3.8-27B work onto one llama.cpp branch carrying every model's patches, and added integer-dot expert
matvecs for IQ2_XXS / IQ2_XS / IQ2_S / IQ3_XXS / IQ3_S and a draft vocabulary for DFlash drafters. **One build now serves all
four llama.cpp models:** [patches/llama.cpp-halo/](patches/llama.cpp-halo/) (46 patches on `529ac893e`) is the recommended
build. Same session against each model's previous build and config, on the 27B harness (chat at temperature 1.0, code and
edits greedy); every adopted config passed the gates, KLD unchanged within noise.

| Model | Previous chat / code / edits / decode @32K | **Unified build** | Adopted change | Gate |
|---|---|---|---|---|
| [MiMo-V2.6-Flash](mimo/) | 28.7 / 35.3 / 43.8 / 25.1 | **31.0 / 39.7 / 50.6 / 27.7** | integer-dot experts, 64K DFlash draft vocab | needle 15/15, agentic 5/5 |
| [Laguna-S-2.1](laguna/) | 35.2 / 42.8 / 39.0 / 29.2 | **36.6 / 44.4–45.2 / 79.1** / 29.3–33.5 | 27B verify kernels; `ngram-mod` as an edit profile | needle 15/15, agentic 5/5 |
| [GLM-5.3-Flash](glm/) | 27.8–28.0 / 29.2–29.4 / 29.6–29.8 / 27.1 | **28.6 / 30.1 / 30.0 / 29.9** | merged FA / matvec work at depth | needle 15/15, agentic 5/5 |

Gate per model, again: `ngram-mod` doubles Laguna's edits but cuts MiMo's and GLM's (their drafters already accept edits
well), and even on Laguna its drafts accept only 25–29% in agentic turns, so serve agents with plain DFlash. Two settings
changed with the unified build: Laguna needs `GGML_VK_FA_KV_SLICE_MAX_MB=4096` (else prefill at 128K halves, 311 → 184),
and GLM needs `GGML_VK_DMMV_LARGE_MAX_M=1024 GGML_VK_DMMV_LARGE_MIN_K=8192`. GLM's memory headroom is unchanged
(7–9 GiB free while serving): keep its watchdog.

## Repository layout

- [qwen-strata/](qwen-strata/) — Qwen3.8-Flash-Next on Strata: the v0.1.41 upgrade, the retune, the gap to desktop numbers.
- [laguna/](laguna/) — Laguna-S-2.1 on llama.cpp: block-sparse decode, KV-sliced prefill, fixing DFlash at depth,
  `ngram-mod` for edits, and Talus vs llama.cpp.
- [mimo/](mimo/) — MiMo-V2.6-Flash on llama.cpp: the attention requant, speculation, integer-dot expert matvecs, a DFlash
  draft vocabulary, prefill profile, thinking control.
- [qwen27b/](qwen27b/) — Qwen3.8-27B on llama.cpp: quant, MTP draft vocabulary, prefill-at-depth and verify-round kernels,
  n-gram drafting, the prompt cache, and Talus vs llama.cpp.
- [glm/](glm/) — GLM-5.3-Flash on llama.cpp: non-expert requant, MTP with a draft vocabulary, sparse-attention and IQ2_S / IQ3_S
  kernels, fusions, and a matched ROCm vs Vulkan run.
- [platform/](platform/) — Strix Halo host settings and lessons that apply to any model.
- [patches/llama.cpp-halo/](patches/llama.cpp-halo/) — **the unified series** (46 patches on `529ac893e`): one build for
  Qwen3.8-27B, Laguna, MiMo and GLM. Recommended.
- [patches/llama.cpp/](patches/llama.cpp/) — our llama.cpp patch series (Vulkan + speculation + Laguna/MiMo support).
- [patches/llama.cpp-q27/](patches/llama.cpp-q27/) — the Qwen3.8-27B series (15 patches on `529ac893e`).
- [patches/llama.cpp-glm/](patches/llama.cpp-glm/) — the GLM-5.3-Flash series (23 patches on a newer base).
- [configs/](configs/) — the exact serving configs (paths replaced with `$PLACEHOLDERS`).

## Reproducibility

- **Qwen on Strata**: stock upstream Strata v0.1.41 + documented switches; nothing patched.
- **Laguna / MiMo / GLM / Qwen3.8-27B on llama.cpp**: `529ac893e` + the [unified series](patches/llama.cpp-halo/)
  reproduces our current build exactly; the per-model configs are the `*-halo.env` presets in [configs/](configs/). The
  older per-model series below still reproduce the earlier numbers.
- **Laguna / MiMo on llama.cpp (older series)**: several gains need our [patch series](patches/llama.cpp/) (block-sparse decode, KV-sliced
  prefill, the DFlash drafter window, adaptive speculation pause, the quantize fix). Each model page marks which settings
  work on stock llama.cpp and which need the patches.
- **GLM on llama.cpp (older series)**: `efe3626b1` + the [GLM series](patches/llama.cpp-glm/) reproduces our build exactly. On stock
  llama.cpp, MTP works, but the sparse-attention, IQ2_S / IQ3_S, verify and fusion speedups need the patches.

## Community posts

- Strata: benchmark report [Niko1221/Strata#1716](https://github.com/Niko1221/Strata/pull/1716); `--expert-cache auto`
  OOM on unified memory [Niko1221/Strata#1715](https://github.com/Niko1221/Strata/issues/1715)
- Laguna: [poolside/Laguna-S-2.1 discussion #43](https://huggingface.co/poolside/Laguna-S-2.1/discussions/43)
- MiMo: [XiaomiMiMo/MiMo-V2.6-Flash-MOPD discussion #2](https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Flash-MOPD/discussions/2)

## License

Docs and configs: MIT. The llama.cpp patches are derived from llama.cpp (MIT); the coopmat1 flash-attention patches are by
Nathan Wilson ([Nathanw1014/llama.cpp](https://github.com/Nathanw1014/llama.cpp)) and keep his authorship in every series.
