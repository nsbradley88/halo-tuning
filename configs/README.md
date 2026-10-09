# Serving configs

Exact configs from the test machine. Paths are placeholders: `$STRATA_DIR` (Strata v0.1.41 checkout), `$STRATA_DATA`
(packs and MTP draft layer), `$STRATA_VENV`, `$ROCM_SDK` (ROCm 7.14.1 TheRock tarball), `$HOME`. The JSON file is not
shell-expanded: replace the placeholders before use. The llama.cpp `.env` files are presets for a Docker-based launcher
(`MODEL_ARGS` are the llama-server arguments); the environment variables they rely on are listed in each model's README.
