# llama.cpp patch series for Qwen3.8-27B (Strix Halo, Vulkan)

Base: llama.cpp `529ac893e` (master, 2026-10-14). Apply with `git am patches/llama.cpp-q27/*.patch`; build with
`-DGGML_VULKAN=ON`. Applied to `529ac893e` it reproduces our gated build exactly. Speedups are on by default; each has an
environment switch for A/B. Separate from [../llama.cpp/](../llama.cpp/) (Laguna/MiMo) and [../llama.cpp-glm/](../llama.cpp-glm/).

| Patch | What | Gain on Qwen3.8-27B |
|---|---|---|
| 0001, 0006 | `--spec-draft-vocab FILE`: MTP draft head over a token subset (qwen35/qwen35moe), one-row scatter | draft step 4.2 → 2.2 ms; chat +10% |
| 0002–0003 | coopmat1 FA micro-optimizations — **by Nathan Wilson** ([Nathanw1014/llama.cpp](https://github.com/Nathanw1014/llama.cpp)), authorship kept | prefill |
| 0004 | concat with a transposed src1 via the tiled transpose copy (`GGML_VK_CONCAT_TRANSPOSE_DISABLE`) | 573 → 80 ms per ubatch |
| 0005 | KV-slice prefill order for FA, KV head first (`GGML_VK_FA_KV_SLICE`, `GGML_VK_FA_KV_HEAD_INNER`) | prompt @32K +30%, @60K +109% |
| 0007 | 2-row MMVQ for Q5_K/IQ4_XS above 4 columns; verify-shape tests | +1–2% |
| 0008, 0011 | gated delta net reads its state from the cache by ids; Vulkan writes rollback snapshots straight into the cache | verify overhead −half |
| 0009 | integer-dot (q8_1) MMVQ for IQ3_S, IQ3_XXS, IQ2_S, IQ2_XS | IQ3_S 5-col 336 → 248 µs |
| 0010 | MTP head keeps only output rows when masked; opt-in deferred catch-up (`LLAMA_MTP_DEFER=1`) | — |
| 0012 | flat grouped-query FA fold for 2–8-token verify (`GGML_VK_FA_GQA_FLAT=0` to disable) | 32K verify FA 23.9 → 13.0 ms |
| 0013, 0015 | mul_mat_vec up to 16 columns on RDNA3/4 (`GGML_VK_MMV_MAX_COLS`) | wide verify 1.4–2.5× |
| 0014 | draft vocab for DFlash2 drafters | drafter 11.8 → 8.9 ms |

Each kernel patch adds `test-backend-ops` cases at the model's shapes (all pass against CPU).
