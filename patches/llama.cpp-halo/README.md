# llama.cpp patch series: one build for every model (Strix Halo, Vulkan)

Base: llama.cpp `529ac893e` (master, 2026-10-14). This is the **recommended single build**: the Qwen3.8-27B series plus the
Laguna/MiMo and GLM series, ported or deduplicated onto one branch, plus three new patches. Apply with
`git am patches/llama.cpp-halo/*.patch`; build with `-DGGML_VULKAN=ON`. Applied to `529ac893e` it reproduces our gated
build exactly. Qwen3.8-27B, Laguna-S-2.1, MiMo-V2.6-Flash and GLM-5.3-Flash all serve from it; the older
[../llama.cpp/](../llama.cpp/), [../llama.cpp-q27/](../llama.cpp-q27/) and [../llama.cpp-glm/](../llama.cpp-glm/) series
stay for reference.

| Patches | What | From |
|---|---|---|
| 0001–0015 | Qwen3.8-27B: MTP/DFlash2 draft vocabulary, KV-slice prefill, transposed concat, GDN state by ids, integer-dot IQ matvecs, flat grouped-query verify FA, 16-column matvecs | [llama.cpp-q27](../llama.cpp-q27/) 0001–0015, unchanged |
| 0016 | DFlash Laguna drafter, layer-input taps, decode FA override | Laguna/MiMo 0001 |
| 0017 | adaptive speculation pause (`LLAMA_SPEC_ADAPTIVE`), opt-in GPU input layer | Laguna/MiMo 0002 |
| 0018 | 32-wide subgroup for coopmat1 FA — **by Nathan Wilson** ([Nathanw1014/llama.cpp](https://github.com/Nathanw1014/llama.cpp)), authorship kept | Laguna/MiMo 0005 |
| 0019 | drafter-only adaptive pause, drafter context cap, rotation knobs | Laguna/MiMo 0006 |
| 0020 | GQA decode kernel, decode knobs, sparse FA on quantized caches (`GGML_VK_FA_SPARSE_MIN_RATIO_Q`) | Laguna/MiMo 0007 (its KV-slice part replaced by 0005) |
| 0021 | Quest-style block-sparse decode attention (`LLAMA_SPARSE_ATTN`) | Laguna/MiMo 0008 |
| 0022 | DFlash drafter window (`DFLASH_SWA`) | Laguna/MiMo 0009 |
| 0023 | MiMo fixes: sparse graph budget, quantize copy-through | Laguna/MiMo 0010 (= GLM 0002) |
| 0024, 0026 | sparse FA for GLM DSA decode / verify / prefill; skips the flat verify fold when the mask is sparse | GLM 0001, 0004 |
| 0025 | imatrix on split GGUFs with NextN layers | GLM 0003 |
| 0027 | `--spec-draft-ubatch` as an alias of upstream `--spec-draft-n-ubatch` | replaces GLM 0006 |
| 0028 | int8 coopmat MMQ for IQ2_S / IQ3_S | GLM 0007 |
| 0029–0036 | GLM graph cuts, KDA state by ids (on 0008's machinery), fusions, mHC pre-mix, decode indexer, sparse FA kernel | GLM 0008–0015 |
| 0037, 0040 | multi-column matvecs merged onto the 16-column matvec; RDNA3 workgroup rule as opt-in knobs (`GGML_VK_DMMV_LARGE_MAX_M` / `_MIN_K`) | GLM 0016, 0018, 0020 |
| 0038 | GLM MTP draft vocabulary through 0001's machinery | GLM 0017 |
| 0039, 0041–0043 | GLM perf test, KDA prologue in the GDN op, split-K `hc_fn`, `LLAMA_N_RS_SEQ` | GLM 0019, 0021–0023 |
| **0044** | integer-dot MMVQ for IQ2_XXS; integer-dot MUL_MAT_ID matvec for IQ2_XXS / IQ2_XS / IQ2_S / IQ3_XXS / IQ3_S experts (`GGML_VK_NO_IQ_ID_MMVQ=1` to disable) | new |
| **0045** | draft vocabulary for DFlash1 drafters (`--spec-draft-vocab`; MiMo's drafter has its own 152K head) | new |
| **0046** | 4 rows per workgroup for the IQ2/IQ3 MUL_MAT_ID MMVQ (`GGML_VK_IQ_ID_ROWS`) | new |

Dropped as duplicates: Laguna/MiMo 0003/0004 (= q27 0002/0003, also by Nathan Wilson) and 0011, GLM 0002, 0005 (covered by
q27 0004) and GLM's own draft-vocab scatter and GDN op copies.

Two defaults changed against the per-model series, set them in the environment:

- `GGML_VK_FA_KV_SLICE_MAX_MB` (KV-slice scratch cap) defaults to 512. At Laguna's ub 4096 that halves prefill at 128K
  (311 → 184 tok/s); `GGML_VK_FA_KV_SLICE_MAX_MB=4096` restores it.
- The RDNA3 matvec workgroup rule (GLM 0018) is off by default; GLM sets `GGML_VK_DMMV_LARGE_MAX_M=1024 GGML_VK_DMMV_LARGE_MIN_K=8192`.

Integer-dot MUL_MAT_ID matvec (0044 + 0046), `test-backend-ops perf`, 256/288 experts, 8 used, 5 tokens, µs (gate-up / down):

| Type | Float path | **MMVQ, 4 rows** |
|---|---|---|
| IQ2_XXS | 508 / 511 | **371 / 375** (−27%) |
| IQ3_XXS | 576 / 582 | **504 / 518** (−12%) |
| IQ2_XS | 489 / 505 | **391 / 408** (−20%) |
| IQ2_S | 522 / 569 | **434 / 478** (−16%) |
| IQ3_S | 609 / 624 | 592 / 601 (−3%) |

Tests (`test-backend-ops` vs CPU, Vulkan): FLASH_ATTN_EXT 5497/5497, MUL_MAT 2665/2665, MUL_MAT_ID 1016/1016,
GATED_DELTA_NET 43/43, GATED_DELTA_NET_IDS_CACHE 39/39, CONCAT, CPY, GLM5_CONV_QKV, the fused ops, LIGHTNING_INDEXER,
GET_ROWS, SET_ROWS, TOP_K and SOFT_MAX all pass.
