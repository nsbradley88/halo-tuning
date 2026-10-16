# Qwen3.8-Flash-Next on Strata (Strix Halo laptop, 70 W)

Model: Qwen3.8-Flash-Next (125B total / ~6B active MoE, DeltaNet hybrid, MTP head), Unsloth **UD-IQ4_XS** GGUF.
Engine: [Strata](https://github.com/Niko1221/Strata) **v0.1.41**, built for gfx1151 against ROCm 7.14.1 (TheRock tarball),
following upstream's `docs/STRIX_HALO.md`. Serving config: [configs/qwen-strata/](../configs/qwen-strata/).

## Final numbers

Engine-level (`--stats`), production config, same prompts (code, 256 new tokens), 112 GiB GTT + `amd_iommu=off`:

| Context | Prompt tok/s | Output tok/s |
|---|---|---|
| 8K | 954–1,009 | **58.5–59.5** |
| 32K | 1,041–1,067 | **57.1–58.6** |
| 128K | 1,005–1,043 | **40.4–44.6** |

Quality: agentic suite **5/5**, needle **15/15** (~83K-token prompts, ~95 s each).

## How we got here

| Step | Prompt @32K | Output @32K | Notes |
|---|---|---|---|
| Our own gfx1151 bring-up (Strata Oct 4 + local commits, ROCm 10.2 nightly) | 705 | 39.4 | tuned hipBLASLt table, WMMA prompt attention, MTP `--spec 4` |
| Upstream **v0.1.41**, defaults only | ~580 (8K) | ~50 (8K) | gfx1151 defaults are on automatically; prompt is slower until the fast config is set |
| v0.1.41 + upstream's documented **fast config** | 983 | 51.8 | `STRATA_PF_FUSED` (prompt experts on the matrix cores) etc., see below |
| + retune: **`--spec 5 --mtp-window 8192`** | ~980 | 54.3–54.7 | +3–9% output at 8K–32K, break-even at 128K |
| + host: 112 GiB GTT, `amd_iommu=off` | **1,041–1,067** | **57.1–58.6** | needs a reboot |

Upgrading from our own bring-up to upstream's release was the single biggest step (+28–41% prompt, +13–32% output). Our
early tuning ran on a base that upstream had already overtaken two days later; check the engine's latest release and
hardware docs before a tuning round.

### The fast config (upstream docs/STRIX_HALO.md §5)

```sh
export STRATA_HIPBLASLT_TUNING=$STRATA_DIR/tools/hip/gfx1151-hipblaslt-100401.txt   # ROCm 7.14.1's hipBLASLt
export STRATA_PF_FUSED=1 STRATA_PF_GEMM=1 STRATA_HC_UPMIX=1 STRATA_PA_FAST=1 STRATA_HIP_WMMA=1 \
       STRATA_SELECT_WMMA=1 STRATA_HC_Q8=1 STRATA_PF_SWITCH_MIN_T=4096 STRATA_PREFILL_STREAM_MIN=128
```

These change rounding slightly (upstream gates them with KL); we re-gated with the agentic suite and needle.

### Engine arguments

`--expert-cache 24576 --resident-budget-gib 55 --prefill auto:16384 --spec 5 --spec-min-p 0.5 --mtp <mtp/rt>
--mtp-window 8192 --max-context 262144 --kv int8` (all experts on the GPU: 55.4 GiB).

## What we tested and did not adopt

| Change | Result |
|---|---|
| `--spec 3` / `--spec 6` | 3: slightly worse; 6: +4% at 8K but **−16% at 32K** |
| `--spec-min-p 0.3 / 0.7`, `--mtp-q4 all`, `--lookup-chain 3`, `--expert-cache-per-layer`, `--prefill auto:8192 / 32768` | no effect beyond noise |
| Upstream's benchmark flags (`--prefill 16384 --spec 4 --lookup-chain 3 --mtp-q4 all`) | same prompt speed, −11% output at 32K |
| A pack rebuilt by v0.1.41's `iq_pack.py --compat-bf16` | identical to our older pack (a pack without `--compat-bf16` is refused for this file; `STRATA_HC_Q8=1` already reads the Q8_0 projections natively) |
| The same v0.1.41 built on ROCm 10.2 instead of 7.14.1 | within noise: the runtime change cost nothing |
| GPU perf level `high` | nothing (the 70 W package limit holds) |
| Batch slots (`"parallel": 2 / 4`, `--batch-mtp`), 4 concurrent clients | total 27.8–30.6 tok/s vs **32.7** one-at-a-time, and the first answer at 31–43 s vs 3.9 s |
| Thinking off for agentic work (`reasoning_effort: none` server-wide) | agentic **4/5**: one task looped for 52 min / 560 tool calls without a fix (thinking on solves it in 90–164 s); needle 15/15. Keep thinking on for Qwen |

## Warning: the hipBLASLt tuning table must match the runtime's hipBLASLt

`STRATA_HIPBLASLT_TUNING` names a table built for one hipBLASLt version: `-100401` is ROCm 7.14.1, `-100500` is the
ROCm 10.2 nightlies, and `tools/hip/` ships both. If you run a build linked against another ROCm with the 7.14.1 table,
the engine rejects it and runs every prompt GEMM untuned. It reports this only once in the engine log:
`hipBLASLt version mismatch: file=100401 runtime=100500; using hipBLASEx`. On a 10.2-nightly build this cut prompt speed
from 716 to ~300 tok/s at ~1.9K and from 1,074 to 829 at ~28K. Output speed was unaffected. With the matching table it
returned to 720 / 1,078. Check the engine log for `hipBLASLt tuning enabled` after any ROCm change.

## Warning: `--expert-cache auto` on a 128 GB unified-memory box

Upstream setup's default (`--expert-cache auto`, no `--resident-budget-gib`) sized the expert cache from free RAM as if it
were spare VRAM and triggered a **global OOM** (the 53 GB engine and processes in the desktop session were killed). Keep an
explicit `--expert-cache` / `--resident-budget-gib` on Strix Halo and test with a memory watchdog.

## The remaining gap to ~1,300 tok/s prompt

Config, pack, ROCm runtime and host boot settings were all A/B'd (median of 3, interleaved): none closes the last ~20%.
A power/clock trace shows why: the laptop holds **70 W** and the GPU averages ~2,070 MHz under prompt processing. Desktop
Strix Halo boxes run at 120 W+. Generation (bandwidth-bound) is unaffected and already matches the published numbers.
