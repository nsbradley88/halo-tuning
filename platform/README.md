# Strix Halo platform notes (any model)

Machine: HP ZBook Ultra G1a 14" (Ryzen AI Max+ 395, Radeon 8060S / gfx1151, 128 GB LPDDR5X), Ubuntu 26.04, kernel 7.0.
Host config snippets: [configs/host/](../configs/host/). llama.cpp ran on Vulkan with a current Mesa RADV driver from the
[strix-halo-llamacpp](https://github.com/Nathanw1014/strix-halo-llamacpp) portable bundle (newer than Ubuntu's Mesa).
Strata ran on ROCm 7.14.1 (AMD TheRock tarball for gfx1151).

## Host settings

| Setting | Why | Effect |
|---|---|---|
| `amdgpu.lockup_timeout=10000,60000,10000,10000` | the default 2 s GPU timeout kills long prompt chunks at 200K+ context ("device lost") | required for long context |
| GTT pool 112 GiB: `ttm.pages_limit=29360128 ttm.page_pool_size=29360128` (was 104 GiB) | room for big models + KV + compute buffers | with `amd_iommu=off`: Strata prompt +8–9%, output +2–8%; llama.cpp prompt +7% |
| `amd_iommu=off` | matches the Strata maintainers' gfx1151 host | see above; **drops DMA protection for USB4/Thunderbolt devices** |
| Platform profile `performance`, on AC | — | the package still holds a hard 70 W |
| `kho=off` (Ubuntu 7.0 kernels) | kexec-handover marks ~8.7 GiB as CMA by default; TTM can't use it | required for GLM-5.3-Flash (~110 GiB): loads above ~116 GiB OOMed or crashed the box without it |
| GTT pool 120 GiB, multi-user target (no GUI): `ttm.pages_limit=31457280` | room for a ~110 GiB model + 64K context + MTP | GLM-5.3-Flash only; ~8–9 GiB stays free, so run it with a MemAvailable watchdog |

## Lessons

- **This laptop is power-limited at 70 W.** Under load the GPU averages ~2,070 MHz of its 2,900 MHz maximum, so
  compute-bound work (prompt processing) runs ~20–25% below desktop Strix Halo numbers; bandwidth-bound decode matches
  them. GPU perf level `high` does not help. Compare prefill numbers only within the same power class.
- **Decode is bandwidth-bound everywhere (~210 GB/s).** Before tuning a model, sum the bytes each generated token reads per
  tensor group (attention, routed experts × top-k / n_expert, dense FFN, output head) and divide by 210 GB/s. If that
  matches the measured speed, only fewer bytes or speculation can help, and the biggest still-high-precision group is
  where a requant pays.
- **Keep ≥ ~20 GiB free with the full serving config loaded.** At ~13 GiB free (MiMo + drafter, 196K context, ub 4096),
  with the desktop also holding GPU buffers, TTM evicted during submission: device lost, then recurring
  `ttm_resource_move_to_lru_tail` list-corruption warnings for every GPU client until a reboot.
- **Strata's `--expert-cache auto` OOMs a 128 GB unified-memory box**: it sizes the cache from free RAM as if it were spare
  VRAM. Set `--expert-cache` / `--resident-budget-gib` explicitly; run tests with a MemAvailable watchdog and launch long
  test scripts with `setsid` so a crashed session does not take them down.
- **Vulkan (llama.cpp): extra per-layer side-chain ops in the forward graph corrupted the forward pass at depth** although
  every op passed `test-backend-ops`; do that bookkeeping on the host after each ubatch.
- **FA KV slicing keeps one partial result per slice, row and head**: with many heads and a large ubatch the scratch can
  exceed `maxStorageBufferRange` (MiMo, 64 heads, ub 4096, slice 4096 at 128K = 4.6 GB → abort). Patch 0011 caps the slice
  count instead.
- **Test speculative decoding greedily** (temperature 0). At temperature 1, runs with and without speculation follow
  different random paths and occasional misreads look like bugs.
- **Agentic wall time is dominated by one long task**: compare aggregate decode tok/s from the server log, not wall time.
- **Check upstream before a tuning round.** We tuned Strata on our own gfx1151 bring-up while upstream shipped official
  gfx1151 support two days later (~1.4–1.8× our prompt speed).
- **A reboot alone can move numbers ~5%**: compare before/after-reboot changes against a same-session baseline.
- **Never compile on the box while benchmarking**: CPU and GPU share the 70 W package budget, and a build halves GPU
  clocks. Sustained load also drifts from ~69 W to ~57 W as the chassis warms over a day, so long A/Bs must alternate runs.
- **ROCm vs Vulkan is a per-build question, not a backend property.** For GLM-5.3-Flash, a matched run (same commit,
  model, flags, session) gave Vulkan 29.6 vs HIP 21.5 tok/s chat and 208 vs 117 tok/s prompt at 32K. The community's
  "ROCm 1.7× faster" result compared against an untuned Vulkan path. Measure your own build before switching.
- **Batching does not pay here**: with prompt-heavy agent turns at 70 W and speculation already making single-stream
  decode efficient, serving several requests at once only delays the first answers.

## Test method

- Speed: `llama-bench -p 0 -n 64 -d <depth>` (decode), `-p 2048/8192 -d <depth>` (prefill at depth); real-server depth
  runs (an edit task at the end of a code context, and a prose task) for speculation; engine `--stats` on Strata.
- Quality, cheapest first: KLD vs a reference quant (`llama-perplexity --kl-divergence`), greedy agreement at depth,
  needle (5 codenames at 10–90% depth of ~81K tokens of code, 3 seeds), agentic suite (5 Go tasks in
  charmbracelet/bubbles via opencode, hidden tests).
- Production stopped during every run; one heavy job at a time; same-session interleaved baselines.
