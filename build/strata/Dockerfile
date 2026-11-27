# Strata (github.com/Niko1221/Strata, MIT) for Inkling 2 Mini on the eGPU laptop's RTX 5090.
# The official CUDA 13 image (Python serve layer + the strata-inkling 0.1.65 shape engine, built for CUDA 13.0 /
# driver 580.65+, so the laptop's R580 is fine) plus two small layers:
#   - patch_inkling_thinking.py: upstream 0.2.19 pins Inkling to reasoning_effort "none" and drops thinking; we keep
#     the request's level (default $STRATA_INKLING_EFFORT=medium) and stream thinking as reasoning_content.
#   - strata-k8s-serve: writes the run config from env and starts serve/server.py directly (no launcher, no TUI,
#     no downloads), bound to 0.0.0.0 for the in-cluster Service.
# Built by .github/workflows/build-strata.yaml.
ARG STRATA_IMAGE=ghcr.io/niko1221/strata:0.2.23-cuda13@sha256:547f07819a53675a7b04f11a9440ad96d75a3218638a83ca0e508ccb293db7d1
FROM ${STRATA_IMAGE}

COPY patch_inkling_thinking.py /opt/strata/patch_inkling_thinking.py
RUN /opt/strata/terminal/python/bin/python3 /opt/strata/patch_inkling_thinking.py /opt/strata/terminal \
    && /opt/strata/terminal/python/bin/python3 -m py_compile /opt/strata/terminal/serve/inkling_dialect.py /opt/strata/terminal/serve/server.py \
    && ls /opt/strata/components/shapes/inkling2/engine/*/strata-inkling

COPY strata-k8s-serve /usr/local/bin/strata-k8s-serve
RUN chmod 0755 /usr/local/bin/strata-k8s-serve

# the base image sets PORT=8080 for its own entrypoint; ours serves the Service port
ENV STRATA_INKLING_EFFORT=medium PORT=8000
EXPOSE 8000
ENTRYPOINT ["/usr/bin/tini", "--", "/usr/local/bin/strata-k8s-serve"]
