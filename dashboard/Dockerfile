# Image Immunizer dashboard — RunPod-ready GPU image.
# Build context must be the repo root (M-Attack-V2/), e.g.:
#   docker build -f dashboard/Dockerfile -t youruser/immunizer:latest .
#
# Torch + CUDA come from the base image; we only add the rest.
FROM pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    WANDB_MODE=disabled \
    WANDB_SILENT=true \
    HF_HOME=/workspace/hf-cache \
    PORT=8000
# NOTE: TARGET_PATH is intentionally NOT set so app.py resolves the target
# relative to wherever it runs (baked /app OR the repo on /workspace/app).

# OS deps: image libs + git (git pulls the app code at runtime via bootstrap).
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 git && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Python deps (torch/torchvision already in base image — torch 2.4.1).
# IMPORTANT: pin transformers to a version compatible with torch 2.4.1.
# The latest transformers requires a newer torch and crashes on import here.
RUN pip install --no-cache-dir \
    "transformers==4.46.3" \
    "tokenizers<0.21" \
    "huggingface_hub<0.26" \
    "numpy<2" \
    open_clip_torch \
    hydra-core \
    omegaconf \
    einops \
    tqdm \
    pyyaml \
    wandb \
    pillow \
    "fastapi>=0.111" \
    "uvicorn[standard]>=0.30" \
    "python-multipart>=0.0.9" \
    "anyio>=4.0"

# Copy the application code (resources/images etc. excluded via .dockerignore).
COPY . /app

# NOTE: models are NOT baked into this image (keeps it ~10GB so it builds on a
# disk-constrained machine). The 4 CLIP surrogates (incl. the large g14) are
# downloaded on the pod's FIRST run and cached there. On RunPod that one-time
# download is fast.
#
# If you later build on a machine/CI with more disk and want instant pod boot,
# uncomment to bake them in:
# RUN python -c "import sys; sys.path.insert(0, '/app'); \
# from surrogates.FeatureExtractors import ClipFeatureExtractor; \
# [ClipFeatureExtractor(n) for n in ['clip_b16','clip_b32','clip_laion_b32','clip_laion_g14']]; \
# print('CLIP weights baked into image')"

EXPOSE 8000

# Launcher: pulls latest code from REPO_URL onto the volume, then runs the app.
# (Falls back to the baked /app code if REPO_URL is unset.)
CMD ["python", "/app/dashboard/bootstrap.py"]
