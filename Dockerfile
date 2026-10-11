# Base image: PyTorch CUDA on Ubuntu 24.04
FROM runpod/pytorch:1.3.2-cu1290-torch291-ubuntu2404

WORKDIR /app

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# System dependencies for video encoding and audio handling
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    git \
    git-lfs \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install uv for fast & compatible builds (ltx-core uses uv_build backend)
RUN pip install --no-cache-dir uv

# Clone LTX-2 repo
RUN git clone --depth 1 https://github.com/Lightricks/LTX-2.git /opt/LTX-2

# Install ltx-core and ltx-pipelines using uv
RUN uv pip install --system --no-cache -e /opt/LTX-2/packages/ltx-core && \
    uv pip install --system --no-cache -e /opt/LTX-2/packages/ltx-pipelines

COPY requirements.txt .
RUN uv pip install --system --no-cache -r requirements.txt

# Model path defaults pointing to persistent Network Volume (/workspace/models)
ENV LTX25_MODELS_DIR=/workspace/models/ltx-2.5
ENV TRANSFORMER_PATH=/workspace/models/ltx-2.5/diffusion_models/ltx-2.5-22b-distilled-transformer-bf16.safetensors
ENV TEXT_ENCODER_PATH=/workspace/models/ltx-2.5/text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors
ENV VIDEO_VAE_PATH=/workspace/models/ltx-2.5/vae/ltx-2.5-video-vae-bf16.safetensors
ENV AUDIO_VAE_PATH=/workspace/models/ltx-2.5/vae/ltx-2.5-audio-vae-bf16.safetensors
ENV SPATIAL_UPSAMPLER_PATH=/workspace/models/ltx-2.5/latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors

COPY handler.py .

CMD ["python", "-u", "handler.py"]
