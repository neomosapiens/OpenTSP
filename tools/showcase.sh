#!/usr/bin/env bash
# The TinyStories showcase in the vLLM environment: `tools/showcase.sh chat|batch|serve [arguments of opentsp_vllm.showcase.cli]`
# (the recipes `just showcase`, `just showcase-batch`, `just showcase-serve`). Environment variables stand in for flags:
#   chat:  PROMPT (one shot), MAX_TOKENS, TEMPERATURE, TOP_K, SEED (a number or `none`), VERBOSE=1
#   batch: N (the number of requests), MAX_TOKENS, TEMPERATURE, TOP_K, SEED
#   serve: PORT
# and OPENTSP_SHOWCASE_DIR names the prepared directory (`just showcase-prepare`).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source "${OPENTSP_TOOLCHAIN:-$HOME/.local/opt/opentsp-toolchain}/env.sh"

mode="${1:?usage: tools/showcase.sh chat|batch|serve [arguments]}"
shift
venv="${OPENTSP_VLLM_VENV:-$HOME/.local/opt/opentsp-vllm-venv}"
if [ ! -x "$venv/bin/python" ]; then
    echo "showcase: the vLLM environment $venv does not exist; run 'just vllm-env' first" >&2
    exit 1
fi
so="opentsp-vllm/native/target/$(uname -s)-$(uname -m)/_tsp_iss.abi3.so"
if [ ! -e "$so" ]; then
    echo "showcase: the ISS binding is not built; run 'just vllm-binding' first" >&2
    exit 1
fi
# Use the same feature selection and installation as the explicit binding recipe.
if ! just vllm-binding if-stale >/dev/null; then
    echo "showcase: error: the ISS binding does not build (see above)" >&2
    exit 1
fi
"$venv/bin/python" tools/check_python_checkout.py "$PWD" >/dev/null

args=()
case "$mode" in
    chat)
        if [ -n "${PROMPT:-}" ]; then args+=(--prompt "$PROMPT"); fi
        if [ -n "${VERBOSE:-}" ]; then args+=(--verbose); fi
        ;;
    batch)
        if [ -n "${N:-}" ]; then args+=(-n "$N"); fi
        ;;
    serve)
        if [ -n "${PORT:-}" ]; then args+=(--port "$PORT"); fi
        ;;
esac
if [ "$mode" != serve ]; then
    if [ -n "${MAX_TOKENS:-}" ]; then args+=(--max-tokens "$MAX_TOKENS"); fi
    if [ -n "${TEMPERATURE:-}" ]; then args+=(--temperature "$TEMPERATURE"); fi
    if [ -n "${TOP_K:-}" ]; then args+=(--top-k "$TOP_K"); fi
    if [ -n "${SEED:-}" ]; then args+=(--seed "$SEED"); fi
fi
exec "$venv/bin/python" -m opentsp_vllm.showcase.cli "$mode" "${args[@]+"${args[@]}"}" "$@"
