# Halo Tuning

Tuning three large MoE models for local inference on an **AMD Strix Halo** (Ryzen AI Max+ 395, Radeon 8060S / gfx1151,
128 GB unified memory): **Qwen3.8-Flash-Next** on [Strata](https://github.com/Niko1221/Strata), and **Laguna-S-2.1** and
**MiMo-V2.6-Flash** on llama.cpp (Vulkan). Every number here was measured on one machine, and every adopted change had to
pass the same quality gates (a long-context needle test and a 5-task agentic coding suite).

| Model | Engine | Output tok/s (short → long context) | Prompt tok/s | Quality |
|---|---|---|---|---|
| [Qwen3.8-Flash-Next](qwen-strata/) (125B-A6B, UD-IQ4_XS) | Strata v0.1.41 | **58.5–59.5** @8K, 57–58.6 @32K, 40–44.6 @128K | ~1,000–1,067 | agentic 5/5, needle 15/15 |
| [Laguna-S-2.1](laguna/) (118B-A8B, Q4 "lean" requant) | llama.cpp Vulkan + [patches](patches/) | 33–34 plain; **code with DFlash 40.6 / 37.5 / 34.0 / 34.9** @1K / 31K / 124K / 235K; 26.4 @128K, 24.2 @245K plain | 611–653 @0, 465 @31K | agentic 5/5, needle 14–15/15 |
| [MiMo-V2.6-Flash](mimo/) (309B-A15B, BPW2.5 + Q4_K attention) | llama.cpp Vulkan + [patches](patches/) | 30.7–32 plain; **code with DFlash 45–47 / 41–43** @1K / 31K | 323–332 @0, ~240 @32K | agentic 5/5, needle 15/15 |

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

| Finding | Qwen | Laguna | MiMo |
|---|---|---|---|
| Speculative decoding is the biggest single decode lever | MTP, `--spec 5 --mtp-window 8192` (+3–9%) | DFlash with a 1024-token drafter window (the "DFlash collapses past 8K" fix) | DFlash, 4-token drafts |
| A requant pays only where non-expert weights dominate the bytes per token | — | no (experts dominate) | **yes**: attention Q6_K → Q4_K, +17% decode |
| Thinking off for agentic work | **no** (one task looped, 4/5) | yes, opt-in (727 s vs 1,632 s; needle 13/15 vs 14/15) | **yes** (923 s vs ~1,900 s typical; needle 15/15) |
| Serving several requests at once (batch slots) | no aggregate gain; first answers much later | no (+3%) | — |
| Block-sparse long-context attention | — | **yes** (+35% decode @128K, +61% @245K) | **no** (breaks recall at the same budget) |

The last row is a warning: a technique that is free on one model can quietly break another. Gate every model separately.

## Repository layout

- [qwen-strata/](qwen-strata/) — Qwen3.8-Flash-Next on Strata: the v0.1.41 upgrade, the retune, the gap to desktop numbers.
- [laguna/](laguna/) — Laguna-S-2.1 on llama.cpp: block-sparse decode, KV-sliced prefill, fixing DFlash at depth.
- [mimo/](mimo/) — MiMo-V2.6-Flash on llama.cpp: the attention requant, speculation, prefill profile, thinking control.
- [platform/](platform/) — Strix Halo host settings and lessons that apply to any model.
- [patches/llama.cpp/](patches/llama.cpp/) — our llama.cpp patch series (Vulkan + speculation + Laguna/MiMo support).
- [configs/](configs/) — the exact serving configs (paths replaced with `$PLACEHOLDERS`).

## Reproducibility

- **Qwen on Strata**: stock upstream Strata v0.1.41 + documented switches; nothing patched.
- **Laguna / MiMo on llama.cpp**: several gains need our [patch series](patches/llama.cpp/) (block-sparse decode, KV-sliced
  prefill, the DFlash drafter window, adaptive speculation pause, the quantize fix). Each model page marks which settings
  work on stock llama.cpp and which need the patches.

## Community posts

- Strata: benchmark report [Niko1221/Strata#1716](https://github.com/Niko1221/Strata/pull/1716); `--expert-cache auto`
  OOM on unified memory [Niko1221/Strata#1715](https://github.com/Niko1221/Strata/issues/1715)
- Laguna: [poolside/Laguna-S-2.1 discussion #43](https://huggingface.co/poolside/Laguna-S-2.1/discussions/43)
- MiMo: [XiaomiMiMo/MiMo-V2.6-Flash-MOPD discussion #2](https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Flash-MOPD/discussions/2)

## License

Docs and configs: MIT. The llama.cpp patches are derived from llama.cpp (MIT); three of them are by Nathan Wilson
([Nathanw1014/llama.cpp](https://github.com/Nathanw1014/llama.cpp)) and keep his authorship.
