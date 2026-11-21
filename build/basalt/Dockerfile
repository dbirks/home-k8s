# Basalt (github.com/jesdga95/basalt, MIT): a C++ inference engine for Blackwell (sm_120) that serves Inkling 2 Mini.
# Slim image: the upstream release archive (built with CUDA 13.0.2) on the matching CUDA runtime, plus the server's venv.
# Upstream's own Dockerfile compiles from source on CUDA 13.4; the laptop's driver is R580 (580.178), so we stay on
# 13.0.x, which the archive was built against. Built by .github/workflows/build-basalt.yaml.
FROM nvidia/cuda:13.0.2-runtime-ubuntu24.04

ARG BASALT_VERSION=v0.5.5
ARG BASALT_SHA256=89da792322248de80d9866dea87f2869fe595ed8311008572405fd3cea5b9c08

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates curl python3 python3-venv \
    && rm -rf /var/lib/apt/lists/*

RUN curl -fsSL -o /tmp/basalt.tgz \
        "https://github.com/jesdga95/basalt/releases/download/${BASALT_VERSION}/basalt-${BASALT_VERSION}-linux-x64.tar.gz" \
    && echo "${BASALT_SHA256}  /tmp/basalt.tgz" | sha256sum -c - \
    && tar xzf /tmp/basalt.tgz -C /opt \
    && mv "/opt/basalt-${BASALT_VERSION}-linux-x64" /opt/basalt \
    && rm -rf /tmp/basalt.tgz /opt/basalt/docs \
    && python3 -m venv /opt/basalt/.venv \
    && /opt/basalt/.venv/bin/pip install --no-cache-dir -r /opt/basalt/requirements.txt \
    && ldd /opt/basalt/bin/basalt | grep -q 'libcublas.so.13 => /' \
    && echo "${BASALT_VERSION}" > /opt/basalt/VERSION

ENV HOME=/data \
    PYTHONUNBUFFERED=1 \
    PATH=/opt/basalt/bin:$PATH

WORKDIR /opt/basalt
EXPOSE 8000
ENTRYPOINT ["basalt", "serve"]
