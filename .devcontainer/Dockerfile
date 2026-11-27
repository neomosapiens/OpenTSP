ARG BASE_IMAGE=ubuntu:24.04
FROM ${BASE_IMAGE}
ARG DEBIAN_FRONTEND=noninteractive
ARG APT_MIRROR
RUN if [ -n "$APT_MIRROR" ]; then sed -i -E "s#http://(archive|security).ubuntu.com/ubuntu#${APT_MIRROR}/ubuntu#g;s#http://ports.ubuntu.com/ubuntu-ports#${APT_MIRROR}/ubuntu-ports#g" /etc/apt/sources.list.d/ubuntu.sources; fi \
    && apt-get update && apt-get install -y --no-install-recommends \
       ca-certificates curl git sudo build-essential cmake ninja-build pkg-config python3.12-venv python3-dev \
       libgmp-dev zlib1g-dev libzstd-dev libnuma-dev libtcmalloc-minimal4t64 libomp5-18 jq rsync time \
    && rm -rf /var/lib/apt/lists/*
RUN usermod -l dev -d /home/dev -m ubuntu && groupmod -n dev ubuntu \
    && echo 'dev ALL=(ALL) NOPASSWD:ALL' > /etc/sudoers.d/dev && chmod 0440 /etc/sudoers.d/dev
USER dev
ENV HOME=/home/dev OPENTSP_TOOLCHAIN=/home/dev/.local/opt/opentsp-toolchain
ARG RUSTUP_DIST_SERVER=https://static.rust-lang.org
ARG RUSTUP_UPDATE_ROOT=https://static.rust-lang.org/rustup
RUN curl -fsSL --retry 3 "${RUSTUP_UPDATE_ROOT}/dist/$(uname -m)-unknown-linux-gnu/rustup-init" -o /tmp/rustup-init \
    && chmod +x /tmp/rustup-init && /tmp/rustup-init -y --no-modify-path --profile minimal --default-toolchain 1.94.0 -c rustfmt -c clippy
ENV PATH=/home/dev/.cargo/bin:/home/dev/.local/opt/opentsp-toolchain/venv/bin:/home/dev/.local/opt/opentsp-toolchain/sail/bin:${PATH}
ARG SAIL_RELEASE_BASE=https://github.com/rems-project/sail/releases/download/0.20.3-binary
ARG PIP_INDEX_URL=https://pypi.org/simple
COPY --chown=dev:dev .devcontainer/toolchain/ /home/dev/.local/opt/opentsp-toolchain/
RUN bash "$OPENTSP_TOOLCHAIN/install-sail.sh" "$OPENTSP_TOOLCHAIN" "$SAIL_RELEASE_BASE" \
    && python3 -m venv "$OPENTSP_TOOLCHAIN/venv" \
    && pip install --no-cache-dir uv rust-just pyyaml mpmath numpy pytest pyflakes huggingface_hub safetensors tokenizers \
    && mkdir -p "$HOME/.cache/opentsp" "$HOME/.cargo/registry" "$HOME/.cargo/git" "$HOME/.cache/uv" "$HOME/.cache/pip"
RUN bash "$OPENTSP_TOOLCHAIN/install-formal.sh" "$OPENTSP_TOOLCHAIN/venv"
USER root
# The numeric-speed gate counts hardware instructions on Linux.
RUN apt-get update && apt-get install -y --no-install-recommends linux-tools-generic libcap2-bin \
    && perf_bin="" && for p in /usr/lib/linux-tools-*/perf; do if [ -x "$p" ]; then perf_bin="$p"; fi; done \
    && test -n "$perf_bin" && ln -sf "$perf_bin" /usr/local/bin/perf \
    && setcap cap_perfmon+ep "$perf_bin" \
    && rm -rf /var/lib/apt/lists/*
USER dev
ENV CUDA_VISIBLE_DEVICES=-1 NVIDIA_VISIBLE_DEVICES=void
WORKDIR /workspaces/opentsp
CMD ["bash"]
