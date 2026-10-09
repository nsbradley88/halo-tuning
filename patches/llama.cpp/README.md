# llama.cpp patch series (Strix Halo, Vulkan)

Base: llama.cpp `a8c9a4e7c` (master, 2026-10-02). Apply with `git am patches/llama.cpp/*.patch`; build with
`-DGGML_VULKAN=ON`. Everything new is opt-in through environment variables (defaults unchanged) except the fixes.

| Patch | What | Used by |
|---|---|---|
| 0001 | Laguna DFlash drafter support (aux norm, gated attention, context norm, causal), layer-input taps | Laguna |
| 0002 | `LLAMA_SPEC_ADAPTIVE=<x>`: pause drafting below x accepted tokens per round | Laguna, MiMo |
| 0003–0005 | coopmat1 FA micro-optimizations — **by Nathan Wilson** ([Nathanw1014/llama.cpp](https://github.com/Nathanw1014/llama.cpp)), authorship kept | all |
| 0006 | adaptive pause for model drafters only, drafter context cap, attention rotation knobs | Laguna, MiMo |
| 0007 | `GGML_VK_FA_KV_SLICE=<tokens>` KV-sliced FA prefill; GQA decode kernel; decode tuning knobs; sparse FA on coopmat1 quantized caches | Laguna, MiMo |
| 0008 | `LLAMA_SPARSE_ATTN=<blocks>,<block>,<min ctx>`: Quest-style block-sparse decode attention (key bounds kept on the host) | Laguna |
| 0009 | `DFLASH_SWA=<tokens>`: sliding window for the DFlash drafter; KV cache room for the key-bound tensor | Laguna |
| 0010 | graph node budget for sparse attention (MiMo); **llama-quantize: no imatrix needed for tensors copied unchanged**; MiMo-shaped test-backend-ops cases | MiMo |
| 0011 | cap the FA KV-slice count to the scratch buffer limit (was an abort) | MiMo |

Notes: block-sparse decode is model-dependent (it broke MiMo's recall at the budget that is free on Laguna); gate it with
a needle test per model. `GGML_VK_FA_WAVE32=1` (from 0005) is set in all our configs.
