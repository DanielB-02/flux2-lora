# FLUX2-LORA-PROJECT.md
> Project context for Claude Code. Read this before making any changes.

---

## What This Project Is

A personal portrait LoRA training pipeline. The goal is to train a FLUX.2 [dev] LoRA on photos of the owner and use it to generate a specific set of target images — realistic portraits in varied lifestyle contexts (street, candid, activity shots). Everything runs on RunPod GPU pods with a persistent network volume. This repo lives on the pod at `/runpod-volume/configs/` and is also cloned locally on the owner's laptop.

---

## Repository Structure

```
flux-lora-workspace/
├── FLUX2-LORA-PROJECT.md          <- this file
├── Dockerfile                     <- custom pod image (built once, pushed to Docker Hub)
├── training_log.md                <- manual log of every training run
├── .gitignore                     <- excludes models, images, outputs
│
├── configs/
│   ├── flux_lora_v1.toml          <- training config (increment version per run)
│   └── flux2_dataset.toml         <- Musubi Tuner dataset config
│
└── scripts/
    ├── prepare_dataset.py         <- crop + resize raw photos to 1024x1024
    ├── caption_dataset.py         <- writes .txt captions alongside training images
    ├── generate.py                <- generate images with FLUX.2 + LoRA (uses Musubi Tuner)
    ├── upload_to_drive.py         <- syncs generated output -> Google Drive via rclone
    └── feedback.py                <- SQLite-based image rating + promotion tool
```

---

## Infrastructure

### RunPod Network Volume — `flux-lora-volume`

Persistent 300GB volume mounted at `/runpod-volume/` on every pod. **Never store anything important only on the pod disk** — it is destroyed when the pod stops.

```
/runpod-volume/
├── models/
│   ├── flux2-dev/              <- FLUX.2-dev weights (~60GB, downloaded once)
│   │   ├── flux2-dev.safetensors   <- BFL single-file checkpoint
│   │   ├── ae.safetensors          <- VAE
│   │   └── text_encoder/           <- Mistral 3 text encoder (sharded)
│   └── loras/                  <- trained LoRA .safetensors outputs
├── datasets/
│   └── myself/
│       ├── raw/                <- original uploaded photos (untouched)
│       └── 10_ohwx man/        <- preprocessed 1024x1024 JPGs + .txt captions + caches
├── outputs/                    <- training checkpoints + TensorBoard logs
│   └── lora_v1/                <- v1 LoRA outputs
├── generated/                  <- generated images from inference
├── configs/                    <- this git repo, cloned here on startup
├── feedback.db                 <- SQLite database for image ratings
└── rclone.conf                 <- Google Drive OAuth config (copied from laptop once)
```

### RunPod Pod

- **Template:** `flux-lora-trainer`
- **Docker image:** `duncanbor/flux-lora-pod:latest`
- **Recommended GPU:** A100 SXM 80GB at ~$1.90/hr (US-KS-2 data center)
- **Fallback GPU:** H100 80GB at ~$2.72/hr
- **Ports:** 6006 (TensorBoard), 22 (SSH)
- **SSH key:** `~/.ssh/id_ed25519`

### Pod Startup Behaviour

The container start command (set in the RunPod template) runs automatically on every boot and does the following in order:

1. Generates SSH host keys and starts sshd
2. Adds the owner's SSH public key to authorized_keys
3. Creates all necessary directories on the network volume
4. Clones this git repo to `/runpod-volume/configs/` if not present, otherwise `git pull`
5. Copies `/runpod-volume/rclone.conf` to `/root/.config/rclone/rclone.conf` if it exists
6. Keeps the container alive

Note: The start command must be a single `bash -c "..."` line with semicolon-separated commands. No shebang, no em-dashes, no multi-line — RunPod's entrypoint breaks on all of these.

### Training & Inference Framework

**Musubi Tuner** (by kohya-ss) — supports FLUX.2-dev natively. Installed on-demand from GitHub since the container disk is ephemeral:

```bash
git clone https://github.com/kohya-ss/musubi-tuner.git /opt/musubi-tuner
cd /opt/musubi-tuner && pip install --break-system-packages -e .
```

Note: sd-scripts does NOT support FLUX.2 (architecture mismatch). Always use Musubi Tuner.

Training command:
```bash
cd /opt/musubi-tuner

# 1. Cache latents (run once per dataset change)
python src/musubi_tuner/flux_2_cache_latents.py \
    --model_version dev \
    --vae /runpod-volume/models/flux2-dev/ae.safetensors \
    --dataset_config /runpod-volume/configs/configs/flux2_dataset.toml

# 2. Cache text encoder outputs (run once per caption change)
python src/musubi_tuner/flux_2_cache_text_encoder_outputs.py \
    --model_version dev \
    --text_encoder /runpod-volume/models/flux2-dev/text_encoder/model-00001-of-00010.safetensors \
    --dataset_config /runpod-volume/configs/configs/flux2_dataset.toml

# 3. Train
accelerate launch --num_cpu_threads_per_process 1 --mixed_precision bf16 \
    src/musubi_tuner/flux_2_train_network.py \
    --model_version dev \
    --dit /runpod-volume/models/flux2-dev/flux2-dev.safetensors \
    --vae /runpod-volume/models/flux2-dev/ae.safetensors \
    --text_encoder /runpod-volume/models/flux2-dev/text_encoder/model-00001-of-00010.safetensors \
    --dataset_config /runpod-volume/configs/configs/flux2_dataset.toml \
    --sdpa --mixed_precision bf16 \
    --timestep_sampling flux2_shift --weighting_scheme none \
    --optimizer_type adamw8bit --learning_rate 1e-4 --gradient_checkpointing \
    --max_data_loader_n_workers 2 --persistent_data_loader_workers \
    --network_module networks.lora_flux_2 --network_dim 32 --network_alpha 16 \
    --max_train_steps 1500 --save_every_n_epochs 1 --seed 42 \
    --lr_scheduler cosine --lr_warmup_steps 100 \
    --output_dir /runpod-volume/outputs/lora_v1 --output_name flux2_lora_v1
```

Inference command:
```bash
cd /opt/musubi-tuner
python src/musubi_tuner/flux_2_generate_image.py \
    --model_version dev \
    --dit /runpod-volume/models/flux2-dev/flux2-dev.safetensors \
    --vae /runpod-volume/models/flux2-dev/ae.safetensors \
    --text_encoder /runpod-volume/models/flux2-dev/text_encoder/model-00001-of-00010.safetensors \
    --lora_weight /runpod-volume/outputs/lora_v1/flux2_lora_v1.safetensors \
    --lora_multiplier 1.0 \
    --prompt "ohwx man, your prompt here" \
    --image_size 1024 1024 --infer_steps 28 --guidance_scale 4.0 \
    --seed 42 --attn_mode torch \
    --save_path /runpod-volume/generated/output_name.png
```

---

## Model Details

| Property | Value |
|----------|-------|
| Base model | FLUX.2 [dev] — 32B parameter flow matching transformer |
| Model path | `/runpod-volume/models/flux2-dev/` |
| HuggingFace repo | `black-forest-labs/FLUX.2-dev` |
| License | FLUX Non-Commercial License |
| LoRA framework | `networks.lora_flux_2` (Musubi Tuner) |
| Training precision | bf16 |
| VRAM required | ~80GB (A100 SXM handles it natively, no quantization needed) |

---

## Training Config

### Musubi Tuner dataset config — `configs/flux2_dataset.toml`

```toml
[general]
resolution = [1024, 1024]
caption_extension = ".txt"
batch_size = 1
enable_bucket = false

[[datasets]]
image_directory = "/runpod-volume/datasets/myself/10_ohwx man"
num_repeats = 10
```

### Key training parameters

| Parameter | Value | Reason |
|-----------|-------|--------|
| `network_dim` | 32 | Higher rank — captures facial detail in 32B model |
| `network_alpha` | 16 | Standard: half of network_dim |
| `learning_rate` | 1e-4 | Proven starting point for person LoRAs on FLUX.2 |
| `max_train_steps` | 1500 | FLUX.2 converges faster than Flux.1; 800-1500 is the sweet spot |
| `lr_scheduler` | cosine | Smooth decay, less likely to overfit than constant LR |
| `timestep_sampling` | flux2_shift | Required for FLUX.2 |
| `weighting_scheme` | none | Recommended for FLUX.2 |
| `gradient_checkpointing` | true | Reduces VRAM usage during training |

**Versioning convention:** when changing training parameters, update `output_name` and `output_dir` to match, and log the run in `training_log.md`.

---

## Dataset

### Trigger Word

`ohwx man` — a nonsense string that won't conflict with existing model concepts. Used at the start of every training caption and in every generation prompt.

### Training Images

Located at `/runpod-volume/datasets/myself/10_ohwx man/`. The folder name encodes the repeat count (10x). Format: `0001.jpg`, `0002.jpg`, ... with matching `0001.txt`, `0002.txt`, ... caption files.

Target dataset size: 20-40 images. Quality matters more than quantity.

### Preprocessing Script — `scripts/prepare_dataset.py`

Takes raw photos from `/datasets/myself/raw/`, center-crops to square, resizes to 1024x1024, saves as JPG. Supports HEIC (iPhone) files.

```bash
python /runpod-volume/configs/scripts/prepare_dataset.py \
    --src /runpod-volume/datasets/myself/raw \
    --dst "/runpod-volume/datasets/myself/10_ohwx man"
```

### Caption Script — `scripts/caption_dataset.py`

Writes a `.txt` caption file alongside each image. Cycles through varied template captions using the trigger word. Always manually review and edit captions after running — accurate captions are the biggest lever on LoRA quality.

```bash
python /runpod-volume/configs/scripts/caption_dataset.py \
    --dir "/runpod-volume/datasets/myself/10_ohwx man" \
    --trigger "ohwx man"
```

---

## Image Generation

### Inference Settings

| Setting | Value |
|---------|-------|
| Steps | 28 (50 for maximum quality) |
| Guidance scale | 4.0 (3.5-4.5 is the sweet spot) |
| Resolution | 1024x1024 |
| Attention mode | torch |
| LoRA multiplier | 1.0 (decrease to 0.8 if artefacts appear) |

### LoRA Strength Guide

| Strength | Effect |
|----------|--------|
| 0.6 | Subtle likeness, more creative freedom |
| 0.8 | Good balance — safe starting point |
| 1.0 | Strong likeness, less variation |
| 1.2+ | May cause artefacts or over-fitting |

---

## Generation Targets

These are the 5 specific images to generate after each training run, plus supporting lifestyle shots. All prompts use trigger word `ohwx man`.

### Target 1 — Primary Portrait
Solo, face clearly visible, at least partial body. Confident, high-value.
```
ohwx man, walking down a city street, wearing a well-fitted dark jacket and
clean trousers, face clearly visible, confident natural expression, soft
golden hour sunlight, shallow depth of field, urban background, candid
lifestyle photography, photorealistic, sharp focus
```

### Target 2 — Contrasting Style Portrait
Solo, different framing to Target 1. If Target 1 is full body -> this is a tighter face shot, and vice versa.
```
ohwx man, sitting at a cafe, soft natural window light falling on face,
slight smile, relaxed confident posture, blurred background, close-up
portrait, wearing a clean fitted shirt, candid atmosphere, photorealistic,
film photography aesthetic
```

### Target 3 — Playing Guitar (Candid)
Eyes on strings, mid-action, not looking at camera.
```
ohwx man, playing electric guitar in a rehearsal room, candid moment,
looking down at the strings, absorbed in the music, dim moody lighting,
amplifiers and cables in background, natural expression, shallow depth
of field, documentary photography style, photorealistic
```

### Target 4 — Reading a Book (Candid)
Genuinely reading, not posing. Eyes on the page.
```
ohwx man, reading a book at a cafe, soft window light, natural absorbed
expression, slight lean forward, coffee cup on table, background slightly
blurred, candid street photography style, warm tones, photorealistic,
sharp focus on face
```

### Target 5 — Nature Scenery (Candid)
Looking at the view, not the camera. Wide shot, environment feels vast.
```
ohwx man, standing on a cliff overlooking dramatic mountain scenery,
looking out at the view, casual outdoor clothing, backlit by golden hour
light, candid moment, wide shot showing environment and figure, epic
landscape photography, photorealistic, sharp
```

### Supporting Lifestyle Shots
Generate 4-8 seeds per prompt, keep the best. Face should be visible even in action shots.
- Jet skiing / water sports
- Skydiving
- Boat with friends (group shots fine here)
- Motorcycle
- Gym / working out
- Sharp outfit on a city street
- Travel / foreign city

---

## Feedback & Rating System — `scripts/feedback.py`

SQLite database at `/runpod-volume/feedback.db`. Stores image path, prompt, seed, LoRA version, rating (1-5), and notes. Images stay as files on disk — the database only stores metadata and paths.

```bash
# Rate images from a session
python /runpod-volume/configs/scripts/feedback.py --session "v1_portraits"

# View summary
python /runpod-volume/configs/scripts/feedback.py --summary

# List images
python /runpod-volume/configs/scripts/feedback.py --list

# Promote 4+ rated images to training dataset
python /runpod-volume/configs/scripts/feedback.py --promote --dry-run
python /runpod-volume/configs/scripts/feedback.py --promote
```

Direct SQLite queries:
```bash
sqlite3 /runpod-volume/feedback.db "SELECT lora_version, AVG(rating), COUNT(*) FROM generations GROUP BY lora_version;"
```

---

## Google Drive Integration — `scripts/upload_to_drive.py`

Uses rclone with OAuth config stored at `/runpod-volume/rclone.conf`. The startup script copies this to `/root/.config/rclone/rclone.conf` automatically on every pod boot.

Remote name: `gdrive`. Destination base: `FluxLoRA/generated/`.

```bash
# Upload with auto-dated folder
python /runpod-volume/configs/scripts/upload_to_drive.py

# Upload with custom folder name
python /runpod-volume/configs/scripts/upload_to_drive.py --folder "v1_portraits"

# Preview without uploading
python /runpod-volume/configs/scripts/upload_to_drive.py --dry-run

# Verify rclone is working
rclone lsd gdrive:FluxLoRA
```

---

## RunPod MCP (Claude Code Integration)

An SSH MCP server is configured in Claude Code to run commands on RunPod pods directly. The connection details change with each pod — update via `claude mcp remove ssh-runpod && claude mcp add ...` with the new IP/port.

---

## Key Conventions

- **Never commit** model weights (`.safetensors`, `.ckpt`, `.bin`), training images, or generated outputs — all in `.gitignore`
- **Version configs** by updating `output_name` and `output_dir` per run
- **Log every run** in `training_log.md` with version, config, LR, steps, rank, and notes
- **All scripts run on the pod**, not locally — paths are hardcoded to `/runpod-volume/`
- **Git is the source of truth** for configs, scripts, and workflows — the pod is ephemeral
- **Install Musubi Tuner on every new pod** — it lives on the container disk which is ephemeral

---

## Common Tasks for Claude Code

### Start a training run
```bash
# 1. SSH into running pod
ssh root@POD_IP -p PORT -i ~/.ssh/id_ed25519

# 2. Fix DNS if needed
echo 'nameserver 8.8.8.8' >> /etc/resolv.conf

# 3. Install Musubi Tuner
git clone https://github.com/kohya-ss/musubi-tuner.git /opt/musubi-tuner
cd /opt/musubi-tuner && pip install --break-system-packages -e .

# 4. Pull latest configs
git -C /runpod-volume/configs pull

# 5. Cache latents + text encoder (if dataset changed)
python src/musubi_tuner/flux_2_cache_latents.py --model_version dev \
    --vae /runpod-volume/models/flux2-dev/ae.safetensors \
    --dataset_config /runpod-volume/configs/configs/flux2_dataset.toml
python src/musubi_tuner/flux_2_cache_text_encoder_outputs.py --model_version dev \
    --text_encoder /runpod-volume/models/flux2-dev/text_encoder/model-00001-of-00010.safetensors \
    --dataset_config /runpod-volume/configs/configs/flux2_dataset.toml

# 6. Run training (use nohup so it survives SSH disconnect)
nohup accelerate launch --num_cpu_threads_per_process 1 --mixed_precision bf16 \
    src/musubi_tuner/flux_2_train_network.py \
    --model_version dev \
    --dit /runpod-volume/models/flux2-dev/flux2-dev.safetensors \
    --vae /runpod-volume/models/flux2-dev/ae.safetensors \
    --text_encoder /runpod-volume/models/flux2-dev/text_encoder/model-00001-of-00010.safetensors \
    --dataset_config /runpod-volume/configs/configs/flux2_dataset.toml \
    --sdpa --mixed_precision bf16 \
    --timestep_sampling flux2_shift --weighting_scheme none \
    --optimizer_type adamw8bit --learning_rate 1e-4 --gradient_checkpointing \
    --max_data_loader_n_workers 2 --persistent_data_loader_workers \
    --network_module networks.lora_flux_2 --network_dim 32 --network_alpha 16 \
    --max_train_steps 1500 --save_every_n_epochs 1 --seed 42 \
    --lr_scheduler cosine --lr_warmup_steps 100 \
    --output_dir /runpod-volume/outputs/lora_v1 --output_name flux2_lora_v1 \
    > /tmp/training.log 2>&1 &
```

### Generate images
```bash
cd /opt/musubi-tuner
python src/musubi_tuner/flux_2_generate_image.py \
    --model_version dev \
    --dit /runpod-volume/models/flux2-dev/flux2-dev.safetensors \
    --vae /runpod-volume/models/flux2-dev/ae.safetensors \
    --text_encoder /runpod-volume/models/flux2-dev/text_encoder/model-00001-of-00010.safetensors \
    --lora_weight /runpod-volume/outputs/lora_v1/flux2_lora_v1.safetensors \
    --lora_multiplier 1.0 \
    --prompt "ohwx man, your prompt here" \
    --image_size 1024 1024 --infer_steps 28 --guidance_scale 4.0 \
    --seed 42 --attn_mode torch \
    --save_path /runpod-volume/generated/output.png
```

### Upload photos for a new training run
```bash
scp -P PORT -i ~/.ssh/id_ed25519 ~/Photos/*.jpg \
    root@POD_IP:/runpod-volume/datasets/myself/raw/

python /runpod-volume/configs/scripts/prepare_dataset.py \
    --src /runpod-volume/datasets/myself/raw \
    --dst "/runpod-volume/datasets/myself/10_ohwx man"

python /runpod-volume/configs/scripts/caption_dataset.py \
    --dir "/runpod-volume/datasets/myself/10_ohwx man" \
    --trigger "ohwx man"
```

### End of session checklist
```bash
# 1. Rate generated images
python /runpod-volume/configs/scripts/feedback.py --session "v1_session"

# 2. Upload keepers to Google Drive
rclone --config /runpod-volume/rclone.conf copy /runpod-volume/generated/ "gdrive:FluxLoRA/generated/"

# 3. Commit any config changes
git -C /runpod-volume/configs add -A
git -C /runpod-volume/configs commit -m "Update configs"
git -C /runpod-volume/configs push

# 4. Stop the pod
```

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| rclone upload failing | `rclone listremotes` — if `gdrive:` missing, rclone.conf not loaded. Check `/runpod-volume/rclone.conf` exists |
| OOM during training | Reduce `batch_size` to 1, ensure `gradient_checkpointing = true`, verify A100 80GB is the selected GPU |
| Generated hands look wrong | Add `detailed hands, correct hand anatomy` to positive prompt |
| Git pull fails on pod | `git -C /runpod-volume/configs fetch --all && git -C /runpod-volume/configs reset --hard origin/main` |
| Training loss not decreasing | Check captions are accurate and varied, verify dataset has 20+ images, try lowering LR to `8e-5` |
| DNS not working on pod | `echo 'nameserver 8.8.8.8' >> /etc/resolv.conf` |
| Musubi Tuner not found | Ephemeral container disk — reinstall: `git clone https://github.com/kohya-ss/musubi-tuner.git /opt/musubi-tuner && cd /opt/musubi-tuner && pip install --break-system-packages -e .` |
| bitsandbytes triton error | `pip install --break-system-packages --upgrade bitsandbytes` |
| SSH connection refused | Pod may have restarted — check new port in RunPod dashboard |
