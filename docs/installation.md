# Build and run OpenTSP

The four components share a Rust workspace, a reference Sail installation and two Python environments. The specification environment builds programs and runs model checks. The serving environment holds CPU vLLM and the editable OpenTSP plugin. All commands below run from the repository root in Bash. Keep the checkpoint, the generated program cache and the two Python environments when updating a checkout.

## Ubuntu development container

Install Docker with Buildx. Open the repository with **Dev Containers: Reopen in Container**, or use the shell launcher:

```bash
tools/devcontainer_build.sh
tools/devcontainer_test.sh test e2e programs model-test model-prove iss-vectors
```

The image uses Ubuntu 24.04, Rust 1.94.0, Python 3.12 and the checksum-verified Sail 0.20.3 binary and Z3 for the target CPU. Post-create builds the reference emulator, workspace binaries and Python binding, and installs the pinned CPU serving package. The shell launcher uses the runc runtime. Rust targets, dependency caches, the serving environment and model programs occupy named volumes. Containers can be removed between runs. When mounting a linked Git worktree, the shell launcher also mounts its common Git directory.

Z3 is separately pinned to 4.15.4; the older solver bundled in the Sail archive is replaced after its own checksum check. For an offline installation, `OPENTSP_Z3_WHEEL` may name the exact pinned wheel; its checksum is still required.

The numeric-speed check inside `model-test` measures actual retired instructions: Darwin uses `/usr/bin/time -l`, and Linux uses `perf stat`. The container includes perf with the `cap_perfmon` file capability, and both launchers grant `PERFMON` with the default seccomp profile. The Docker host must expose a working hardware instruction counter. A VM or hosted runner that does not expose one cannot pass this check; unavailable counters fail explicitly. Hardware availability and remote CI success are separate from installing these tools.

The default build targets the Docker engine's architecture. To select an architecture explicitly set `OPENTSP_DEV_PLATFORM=linux/amd64` or `linux/arm64`. Extra arguments to the build script reach Buildx. `APT_MIRROR`, `PIP_INDEX_URL`, `RUSTUP_DIST_SERVER`, `RUSTUP_UPDATE_ROOT` and `SAIL_RELEASE_BASE` select download mirrors. The Sail digest check also applies to mirrored archives.

## Native Linux

On Ubuntu 24.04 install the build and CPU runtime libraries:

```bash
sudo apt-get update
sudo apt-get install build-essential ca-certificates curl git libgmp-dev zlib1g-dev libzstd-dev pkg-config python3.12-venv python3-dev libnuma-dev libomp5-18 libtcmalloc-minimal4t64 linux-tools-generic libcap2-bin
```

Install Rust 1.94.0 using rustup, including rustfmt and clippy. In a fresh toolchain directory, install the same Sail binary and shell environment as the container:

```bash
rustup toolchain install 1.94.0 --component rustfmt --component clippy
rustup override set 1.94.0
export OPENTSP_TOOLCHAIN="${OPENTSP_TOOLCHAIN:-$HOME/.local/opt/opentsp-toolchain}"
mkdir -p "$OPENTSP_TOOLCHAIN"
cp -R .devcontainer/toolchain/. "$OPENTSP_TOOLCHAIN/"
bash "$OPENTSP_TOOLCHAIN/install-sail.sh" "$OPENTSP_TOOLCHAIN"
python3.12 -m venv "$OPENTSP_TOOLCHAIN/venv"
"$OPENTSP_TOOLCHAIN/venv/bin/pip" install rust-just uv pyyaml mpmath numpy pytest pyflakes huggingface_hub safetensors tokenizers
bash "$OPENTSP_TOOLCHAIN/install-formal.sh" "$OPENTSP_TOOLCHAIN/venv"
source "$OPENTSP_TOOLCHAIN/env.sh"
```

For the Linux retired-instruction check, select the installed perf binary and grant its narrow counter capability:

```bash
perf_bin=""
for p in /usr/lib/linux-tools-*/perf; do if [ -x "$p" ]; then perf_bin="$p"; fi; done
test -n "$perf_bin"
sudo setcap cap_perfmon+ep "$perf_bin"
export PATH="$(dirname "$perf_bin"):$PATH"
perf stat --no-big-num -x, -e instructions -- true
```

The last command must report a positive hardware count. It does not qualify any model property.

## Native macOS on Apple Silicon

Install Xcode Command Line Tools, Homebrew and rustup. Use Homebrew Bash for the setup shell. The GNU utilities supply the command names used by the validation recipes.

```bash
xcode-select --install
brew install bash coreutils gmp zstd pkg-config z3 opam just python@3.12
rustup toolchain install 1.94.0 --component rustfmt --component clippy
rustup override set 1.94.0
opam init
opam switch create sail ocaml-base-compiler.5.1.1
opam install --switch=sail sail.0.20.3
export OPENTSP_TOOLCHAIN="${OPENTSP_TOOLCHAIN:-$HOME/.local/opt/opentsp-toolchain}"
mkdir -p "$OPENTSP_TOOLCHAIN/zwrap"
python3.12 -m venv "$OPENTSP_TOOLCHAIN/venv"
"$OPENTSP_TOOLCHAIN/venv/bin/pip" install uv pyyaml mpmath numpy pytest pyflakes huggingface_hub safetensors tokenizers
```

In a new installation, create the following `env.sh` in that directory. Reuse an existing OpenTSP environment when it already supplies these settings.

```bash
cat >"$OPENTSP_TOOLCHAIN/env.sh" <<'ENV'
export OPENTSP_TOOLCHAIN="${OPENTSP_TOOLCHAIN:-$HOME/.local/opt/opentsp-toolchain}"
eval "$(opam env --switch=sail --set-switch)"
export PATH="$OPENTSP_TOOLCHAIN/zwrap:$OPENTSP_TOOLCHAIN/venv/bin:$(brew --prefix)/opt/coreutils/libexec/gnubin:$(brew --prefix)/bin:$HOME/.cargo/bin:$PATH"
export GMP_PREFIX="$(brew --prefix gmp)" GMP_LIBS="$(brew --prefix gmp)/lib/libgmp.dylib" Z_LIBS=-lz
export PYTHON="$OPENTSP_TOOLCHAIN/venv/bin/python"
ENV
sed 's@"$(dirname "$0")/../sail/bin/z3"@"$(brew --prefix)/bin/z3"@' .devcontainer/toolchain/zwrap/z3 > "$OPENTSP_TOOLCHAIN/zwrap/z3"
chmod +x "$OPENTSP_TOOLCHAIN/zwrap/z3"
bash .devcontainer/toolchain/install-formal.sh "$OPENTSP_TOOLCHAIN/venv"
source "$OPENTSP_TOOLCHAIN/env.sh"
z3 --version
```

`just model-prove` requires Z3 4.15.4. If Homebrew supplies a different version, install the official macOS ARM64 build of 4.15.4 and set the wrapper's executable path to its `z3` binary. Confirm the version again before running proof checks.

## Lean proof prerequisites

`just lean-check` and the native mutation cases additionally require Lean 4.29.0 and Lake. `install-formal.sh` installs the RAS RTL tools; it does not install Lean. Run the pinned installer and toolchain commands below.

```bash
curl -fsSL https://raw.githubusercontent.com/leanprover/elan/v4.4.0/elan-init.sh -o /tmp/opentsp-elan-init.sh
sh /tmp/opentsp-elan-init.sh -y --no-modify-path --default-toolchain leanprover/lean4:v4.29.0
export PATH="$HOME/.elan/bin:$PATH"
lean +leanprover/lean4:v4.29.0 --version
lake +leanprover/lean4:v4.29.0 --version
just lean-check
```

Allow an estimated 2.6 GB for the toolchain and additional space for generated projects and build caches. The first proof run needs access to the pinned `lean-sail` Git revision recorded in the generated project's lockfile. An already installed native Lean/Lake 4.29.0 can supply the same tools; verify their actual versions in the shell running the gate. Tests and component proofs remain useful with [whole-machine assurance deferred](assurance.md); this setup does not change that limitation.

## Build and exercise each component

```bash
source "${OPENTSP_TOOLCHAIN:-$HOME/.local/opt/opentsp-toolchain}/env.sh"
just python-setup
opentsp-arch-spec/tools/build_sim.sh
cargo build --release --workspace
just test
just programs
just asm-diff
just iss-vectors
just iss-diff
just model-test
just model-prove
just iss-bench
just vllm-env
just vllm-binding
just vllm-surface
just vllm-unit
just vllm-e2e
```

The Sail command creates `target/spec/tsp_sim`. An explicit output path selects another location. `target/release/ots-asm` assembles and packages programs, and `target/release/tsp-iss` runs them. `just programs` exercises the assembler and compares execution on both simulators. Run `just ras-prove` to check the serving RAS interface against its Sail contract and prove the management RTL. The setup above supplies Yosys, SymbiYosys and Boolector through [YoWASP](https://yowasp.org/).

The default vLLM setup uses the upstream 0.33.0 CPU wheel. Linux also installs CPU builds of Torch, TorchVision and TorchAudio. For an offline copy of the large wheels, set `OPENTSP_VLLM_WHEEL` to the vLLM wheel and `OPENTSP_TORCH_LINKS` to a directory containing the three Torch wheels. On Linux x86-64 with Python 3.12, `OPENTSP_TRITON_WHEEL` accepts the original Triton CPU wheel and verifies its upstream SHA-256. The other Python dependencies still require a populated package cache or index. Set `OPENTSP_VLLM_FROM_SOURCE=1` to build the pinned vLLM source distribution instead.

## Set up the demo runtime

Follow the [demo guide](demos.md) for checkpoint downloads, program packages and launch commands.

For the container shell launcher, set `OPENTSP_DEV_HF_CACHE="$HOME/.cache/huggingface"` and `OPENTSP_DEV_CACHE="$HOME/.cache/opentsp"` on the host to mount populated caches. The Hugging Face cache is read-only. The OpenTSP cache is writable for generated programs and checkpoint manifests. Run `tools/devcontainer_test.sh showcase-prepare showcase-test` to prepare and test TinyStories in the container.
