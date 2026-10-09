#!/bin/sh
# strata-server.service (via drop-in /etc/systemd/system/strata-server.service.d/upstream.conf): Strata v0.1.41 on :8000
# (all interfaces, API key required). Key from the unit's EnvironmentFile (/etc/vllm/secrets.env, root only).
# gfx1151: ROCm 7.14.1 engine (build-halo), upstream's hipBLASLt table, the documented fast config (docs/STRIX_HALO.md
# section 5; gated here 2026-10-09: agentic 5/5, needle 15/15). Revert: remove the drop-in, daemon-reload, restart.
cd "$STRATA_DIR"
export STRATA_API_KEY="${STRATA_API_KEY:-${LLAMA_API_KEY:-${VLLM_API_KEY:-}}}"
[ -n "$STRATA_API_KEY" ] || { echo "strata-server: no API key in /etc/vllm/secrets.env (STRATA_API_KEY / LLAMA_API_KEY / VLLM_API_KEY)" >&2; exit 1; }
export STRATA_HIPBLASLT_TUNING=$STRATA_DIR/tools/hip/gfx1151-hipblaslt-100401.txt
export STRATA_PF_FUSED=1 STRATA_PF_GEMM=1 STRATA_HC_UPMIX=1 STRATA_PA_FAST=1 STRATA_HIP_WMMA=1 STRATA_SELECT_WMMA=1 STRATA_HC_Q8=1 \
       STRATA_PF_SWITCH_MIN_T=4096 STRATA_PREFILL_STREAM_MIN=128
exec $STRATA_VENV/bin/python $STRATA_DIR/serve/server.py --engine strata \
  --config $STRATA_DIR/strata-service.json --host 0.0.0.0 --port 8000
