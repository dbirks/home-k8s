# ComfyUI for MiniMax-H3 on RTX PRO 6000 Blackwell (SM120).
# The maintained prebuilts (ai-dock) are stale (Nov 2024: ComfyUI 0.2.x, torch cu121)
# and can't run H3 or compute on sm_120 — so we build a small pinned image:
#   CUDA 12.8 base + torch cu128 (Blackwell kernels) + recent ComfyUI (H3 in core).
# Models are NOT baked in — they mount from a PVC via --extra-model-paths-config.
FROM nvidia/cuda:13.3.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PIP_BREAK_SYSTEM_PACKAGES=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
      python3 python3-pip python3-dev git ffmpeg libgl1 libglib2.0-0 \
      ca-certificates wget \
    && rm -rf /var/lib/apt/lists/* \
    && python3 -m pip install --upgrade pip

# Blackwell-capable torch FIRST so ComfyUI's requirements don't pull a non-cu128 build.
RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# Recent ComfyUI (H3 support is in core). Pin via build-arg; default master for day-0 models.
ARG COMFYUI_REF=master
RUN git clone https://github.com/comfyanonymous/ComfyUI /opt/ComfyUI \
    && cd /opt/ComfyUI \
    && git checkout ${COMFYUI_REF} \
    && pip3 install -r requirements.txt \
    && pip3 install "torch>=2.7" torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

WORKDIR /opt/ComfyUI
EXPOSE 8188
# All serve flags come from the Deployment `args`.
ENTRYPOINT ["python3", "main.py"]
