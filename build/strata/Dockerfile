# Strata (github.com/Niko1221/Strata, MIT) for Inkling 2 Mini on the eGPU laptop's RTX 5090.
# The official CUDA 13 image (Python serve layer + the strata-inkling 0.1.65 shape engine, built for CUDA 13.0 /
# driver 580.65+, so the laptop's R580 is fine) plus two small layers:
#   - patch_inkling_thinking.py: upstream 0.2.19 pins Inkling to reasoning_effort "none" and drops thinking; we keep
#     the request's level (default $STRATA_INKLING_EFFORT=medium) and stream thinking as reasoning_content.
#   - strata-k8s-serve: writes the run config from env and starts serve/server.py directly (no launcher, no TUI,
#     no downloads), bound to 0.0.0.0 for the in-cluster Service.
# Built by .github/workflows/build-strata.yaml.
ARG STRATA_IMAGE=ghcr.io/niko1221/strata:0.2.19-cuda13@sha256:5cf423c2ad9d2f626da250ade74cb9b48087279bff636fca3c7c08eb4f786d26
FROM ${STRATA_IMAGE}

COPY patch_inkling_thinking.py /opt/strata/patch_inkling_thinking.py
RUN /opt/strata/terminal/python/bin/python3 /opt/strata/patch_inkling_thinking.py /opt/strata/terminal \
    && /opt/strata/terminal/python/bin/python3 -m py_compile /opt/strata/terminal/serve/inkling_dialect.py /opt/strata/terminal/serve/server.py \
    && ls /opt/strata/components/shapes/inkling2/engine/*/strata-inkling

COPY strata-k8s-serve /usr/local/bin/strata-k8s-serve
RUN chmod 0755 /usr/local/bin/strata-k8s-serve

ENV STRATA_INKLING_EFFORT=medium
EXPOSE 8000
ENTRYPOINT ["/usr/bin/tini", "--", "/usr/local/bin/strata-k8s-serve"]
