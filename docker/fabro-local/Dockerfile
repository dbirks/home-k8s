# Fabro server + the tools its `local` sandbox provider needs (issue #161).
# Upstream's server image is minimal Alpine with no /bin/bash, but Fabro runs every sandbox
# command through bash ("sandbox-driver-sentinel: bash: not found"), so agent nodes fail on
# `local`. This adds bash plus the basics from upstream's docker/Dockerfile.agent.
# Bump FABRO in lockstep with apps/fabro.yaml (tag AND digest).
ARG FABRO=ghcr.io/fabro-sh/fabro:0.403.0-nightly.0@sha256:ce754f5b12162a7b9f861934051113a6e8781af3b4522d8243ec3daaade09a16
FROM ${FABRO}
USER root
RUN apk add --no-cache bash coreutils findutils grep sed gawk diffutils patch curl ripgrep \
      jq python3 make openssh-client
LABEL org.opencontainers.image.source=https://github.com/dbirks/home-k8s
