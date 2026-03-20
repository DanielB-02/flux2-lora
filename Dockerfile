FROM runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404

RUN apt-get update && apt-get install -y \
    git wget curl tmux vim rsync ffmpeg \
    libgl1 libglib2.0-0 \
    libsm6 libxrender1 libxext6 \
    rclone \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    diffusers>=0.37.0 \
    transformers>=4.56.0 \
    accelerate>=1.6.0 \
    peft>=0.18.0 \
    bitsandbytes>=0.45.0 \
    sentencepiece \
    huggingface_hub \
    safetensors \
    tensorboard \
    toml \
    Pillow==10.4.0 \
    tqdm \
    einops \
    lion-pytorch \
    prodigyopt \
    opencv-python \
    aiohttp \
    aiofiles \
    yarl \
    torchaudio \
    kornia \
    spandrel \
    soundfile \
    scipy/

# Musubi Tuner — FLUX.2-native LoRA training framework (sd-scripts does NOT support FLUX.2)
RUN git clone https://github.com/kohya-ss/musubi-tuner.git /opt/musubi-tuner && \
    cd /opt/musubi-tuner && pip install --no-cache-dir -e .

WORKDIR /workspace
RUN echo 'echo "=== Flux LoRA Pod Ready ==="' >> /root/.bashrc