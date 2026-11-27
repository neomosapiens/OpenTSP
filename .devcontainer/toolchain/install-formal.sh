#!/usr/bin/env bash
# Install the tools used by the RAS RTL proof into the specification Python environment.
set -euo pipefail
bin="${1:?specification venv directory}/bin"
"$bin/python" -m pip install yowasp-yosys==0.71.0.0.post1285 yowasp-boolector==3.2.3.6.post31.dev0
for tool in yosys sby yosys-smtbmc boolector; do
    ln -sf "yowasp-$tool" "$bin/$tool"
done
# Compile the solver before its first SMT session, where startup output would break the protocol.
"$bin/boolector" --version
