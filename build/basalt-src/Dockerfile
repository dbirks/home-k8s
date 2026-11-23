# Basalt (github.com/jesdga95/basalt, MIT) built FROM SOURCE at a pinned upstream commit: the same C++/CUDA engine as
# build/basalt/Dockerfile, but from `main` instead of a release archive, so we can run unreleased fixes (the tool-call
# parser #540, idle-vision VRAM, the prefill speedups, Inkling suffix drafts) without waiting for a tag.
#
# Why CUDA 13.0.2 and not upstream's 13.4.3 default (build --build-arg CUDA_TAG=13.4.3): this image runs on the laptop's
# RTX 5090 over Thunderbolt (talos-8me-tcy), whose driver is R580 (580.178) = CUDA 13.0. A runtime newer than the driver
# branch will not load; 13.0.2 is what build/basalt/Dockerfile already ships and what apps/inkling2-mini-basalt.yaml
# boots today. The only CUDA libraries the engine links are cudart and cuBLAS, both from the base image.
#
# Why BASALT_MODELS=inkling2: a build left at `all` compiles every model's C++/CUDA sources, kernels and tests, which
# docs/building.md calls "most of the build time for DeepSeek". This image only ever serves Inkling 2 Mini, and a pack
# of a model the build left out refuses to load with a clear message, so inkling2-only costs us nothing here. Pass
# --build-arg BASALT_MODELS="qwen4exp;inkling2" to serve another engine from the same image.
#
# The tree inside the image is a release archive's — bin/, serve/, tools/, data/, .venv/ under /opt/basalt — so the
# runtime contract (ENV, WORKDIR, ENTRYPOINT) is identical to build/basalt/Dockerfile and
# apps/inkling2-mini-basalt.yaml's `basalt serve PACK ...` args work unchanged. The engine finds its install root as
# the directory above the binary, reads expert profiles from <root>/data/profiles, execs <root>/.venv/bin/python for
# `basalt serve`, and looks for the basalt-vision sidecar beside itself. The venv's scripts name the install path
# absolutely, so the build happens at /opt/basalt — the same path it runs from.
#
# Mounts (same as upstream / build/basalt): /data is HOME (the engine log and, when on, the conversation disk tier under
# ~/.cache/basalt), /models holds the .basalt packs.
#
# Built by .github/workflows/build-basalt-src.yaml:
#   docker build --build-arg BASALT_REF=<40-char sha> -t ghcr.io/dbirks/basalt:main-<short> build/basalt-src
# No GPU is needed to build: the engine compiles for sm_120 with no device present and nothing runs CUDA during the
# build. Upstream's own Dockerfile COPYs a local checkout; we clone and pin instead, which is the whole point here.

ARG BASALT_REF
ARG CUDA_TAG=13.0.2
ARG BASALT_MODELS=inkling2

# ------------------------------------------------------------------ build stage
FROM nvidia/cuda:${CUDA_TAG}-devel-ubuntu24.04 AS builder

ARG BASALT_REF
ARG BASALT_MODELS

# git and cmake are all that is needed on top of the devel image's toolchain: the build fetches nothing.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates cmake git python3 python3-venv \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/basalt

# Pin first, then verify: a short sha, a sha git cannot reach, or a ref that moved all stop the build here rather than
# shipping an image whose commit nobody can name.
RUN set -eux; \
    echo "$BASALT_REF" | grep -Eq '^[0-9a-f]{40}$'; \
    git clone https://github.com/jesdga95/basalt.git /opt/basalt; \
    git -C /opt/basalt checkout --detach "$BASALT_REF"; \
    test "$(git -C /opt/basalt rev-parse HEAD)" = "$BASALT_REF"

# The sm_120 engine (basalt, basalt-bench, basalt-probe) plus the basalt-vision sidecar, into /opt/basalt/bin, and the
# server's venv — upstream's steps and paths.
RUN set -eux; \
    cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DBASALT_MODELS="$BASALT_MODELS"; \
    cmake --build build -j"$(nproc)"; \
    python3 -m venv .venv; \
    .venv/bin/pip install --no-cache-dir -r requirements.txt; \
    test -x bin/basalt; \
    test -x bin/basalt-vision

# ----------------------------------------------------------------- final stage
FROM nvidia/cuda:${CUDA_TAG}-runtime-ubuntu24.04

ARG BASALT_REF

# The base already carries the CUDA runtime and cuBLAS, the only CUDA libraries the engine links. python3 is what the
# venv's python points at.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates python3 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/basalt

# The runtime tree: the binaries, the server and the tools it imports, the data/ the engine reads, the venv, licences.
COPY --from=builder /opt/basalt/.venv            /opt/basalt/.venv
COPY --from=builder /opt/basalt/bin              /opt/basalt/bin
COPY --from=builder /opt/basalt/serve            /opt/basalt/serve
COPY --from=builder /opt/basalt/tools            /opt/basalt/tools
COPY --from=builder /opt/basalt/data             /opt/basalt/data
COPY --from=builder /opt/basalt/requirements.txt \
                  /opt/basalt/requirements.txt
COPY --from=builder /opt/basalt/LICENSE \
                  /opt/basalt/THIRD_PARTY_LICENSES \
                  /opt/basalt/

# No GPU here, and none is needed: ldd proves the CUDA 13 libraries resolve against this base, and `basalt serve
# --help` is intercepted before any CUDA call and execs the venv's python, so it also proves the venv, the serve/ tree
# and requirements.txt all work. (There is no `basalt --version`: the version lives in the binary's stamp and in
# /opt/basalt/VERSION.) HOME is overridden because /data is a mount that does not exist at build time.
RUN set -eux; \
    echo "$BASALT_REF" > /opt/basalt/VERSION; \
    test -s /opt/basalt/VERSION; \
    ldd /opt/basalt/bin/basalt | grep -q 'libcublas.so.13 => /'; \
    ldd /opt/basalt/bin/basalt | grep -q 'libcudart.so.13 => /'; \
    HOME=/tmp timeout 300 /opt/basalt/bin/basalt serve --help 2>&1 | grep -qi 'usage:'

ENV HOME=/data \
    PYTHONUNBUFFERED=1 \
    PATH=/opt/basalt/bin:$PATH

# The deployment passes --port 8000 explicitly; upstream's image exposes its own default of 8080.
WORKDIR /opt/basalt
EXPOSE 8000
ENTRYPOINT ["basalt", "serve"]
