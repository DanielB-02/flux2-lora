FROM runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404

RUN apt-get update && apt-get install -y \
    git wget curl tmux vim rsync ffmpeg \
    libgl1 libglib2.0-0 \
    libsm6 libxrender1 libxext6 \
    rclone \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    diffusers==0.30.3 \
    transformers==4.44.2 \
    accelerate==0.34.2 \
    peft==0.12.0 \
    bitsandbytes==0.44.0 \
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
    scipy

# Kohya sd-scripts — Flux LoRA training framework
RUN git clone https://github.com/kohya-ss/sd-scripts /opt/sd-scripts && \
    cd /opt/sd-scripts && git checkout main && \
    pip install --no-cache-dir -r requirements.txt

WORKDIR /workspace
RUN echo 'echo "=== Flux LoRA Pod Ready ==="' >> /root/.bashrc