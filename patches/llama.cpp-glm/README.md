# llama.cpp patch series for GLM-5.3-Flash (Strix Halo, Vulkan)

Base: llama.cpp `efe3626b1` (master, 2026-10-12). This is a separate series from [../llama.cpp/](../llama.cpp/) (Laguna /
MiMo, older base). Apply with `git am patches/llama.cpp-glm/*.patch`; build with `-DGGML_VULKAN=ON`. Applied to `efe3626b1`
it reproduces the production build exactly. Speedups are on by default; each has an environment switch to turn it off for
A/B tests. The fused ops fall back to the unfused graph on backends without them (e.g. HIP).

| Patch | What | Gain on GLM |
|---|---|---|
| 0001 | Vulkan: fold GLM's 64 query heads per KV head as 2 × 32 in FA; per-row sparse FA for small batches (DSA decode, MTP verify) | 32K decode with MTP 13 → 21 tok/s; FA 6.9 → 2.8 ms per token at 32K |
| 0002 | llama-quantize: require an imatrix only for tensors that change type | enables the requant |
| 0003 | llama-imatrix: split GGUFs with NextN layers, source-less graph ops | enables our imatrix |
| 0004 | Vulkan: sparse FA for prefill batches | 60K prompt 86 → 128 tok/s (with 0005) |
| 0005 | glm5-next: contiguous transpose before the KDA conv concat (was 23 ms per call) | prefill |
| 0006 | `--spec-draft-ubatch N`: smaller compute buffer for the draft context (upstream now: `--spec-draft-n-ubatch`, #30459) | memory → ub 1024 fits |
| 0007 | Vulkan: int8 coopmat MMQ for IQ2_S and IQ3_S, dense and mul_mat_id | prompt 2K 172 → 203 |
| 0008 | glm5-next: graph-level dispatch cuts for decode and MTP verify | fusion group, ~+7–10% total |
| 0009 | gated delta net: KDA state read from the cache by ids, Vulkan writes snapshots back in place | (fusion group) |
| 0010 | Vulkan: fuse RMS_NORM + SCALE and SCALE → SIGMOID → SCALE | (fusion group) |
| 0011 | glm5-next: one shared −inf row repeated into the indexer scatter mask | memory |
| 0012–0013 | fused mHC pre-mix op (`DSV4_HC_PRE_MIX`), CPU + Vulkan; vec4 Vulkan path | −990 dispatches per token, +3–5% chat |
| 0014 | Vulkan: decode lightning-indexer kernel for small batches | 32K decode +7% (with 0015) |
| 0015 | Vulkan: dedicated sparse FA kernel for DSA (decode, verify, prefill) | 32K prompt +10% |
| 0016 | Vulkan: batched multi-column mul_mat_vec, strided B, AMD large-workgroup knobs; GLM verify-shape tests | verify batches |
| 0017 | `--spec-draft-vocab FILE`: MTP draft head over a token subset (output stays exact) | 32K decode +10–14% (with 0016/0018) |
| 0018 | Vulkan: RDNA3 mul_mat_vec workgroup rule (four subgroups for few-row / long-row matrices) | verify batches |
| 0019 | test-backend-ops: GLM matvec perf case after a dirty copy | test only |
| 0020 | Vulkan: read strided B in place only for rows ≥ 512 elements | fix for 0016 |
| 0021 | glm5-next: KDA prologue inside the gated-delta-net op, split-K `hc_fn` matvec | ~+1% |
| 0022 | Vulkan: `hc_fn` split-K partials read contiguous row slices; parallel slice sum | (with 0021) |
| 0023 | `LLAMA_N_RS_SEQ`: test knob to force the recurrent rollback depth | test only |

Kill switches for A/B tests: `GGML_VK_FA_SPARSE_ROWS`, `GGML_VK_FA_GQA_SPLIT_DISABLE`, `GGML_VK_FA_SPARSE_PREFILL_RATIO`,
`LLAMA_HC_PRE_MIX=0|nofold`, `LLAMA_HC_MIX_SPLIT=0`, `LLAMA_KDA_PROLOGUE=0`. Each kernel patch adds `test-backend-ops`
cases at GLM's shapes.
