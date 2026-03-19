# FLUX2-LORA-PROJECT.md
> Project context for Claude Code. Read this before making any changes.

---

## What This Project Is

A personal portrait LoRA training pipeline. The goal is to train a FLUX.2 [dev] LoRA on photos of the owner and use it to generate a specific set of target images in ComfyUI — realistic portraits in varied lifestyle contexts (street, candid, activity shots). Everything runs on RunPod GPU pods with a persistent network volume. This repo lives on the pod at `/runpod-volume/configs/` and is also cloned locally on the owner's laptop.

---

## Repository Structure

```
flux-lora-workspace/
├── FLUX2-LORA-PROJECT.md          ← this file
├── Dockerfile                     ← custom pod image (built once, pushed to Docker Hub)
├── training_log.md                ← manual log of every training run
├── .gitignore                     ← excludes models, images, outputs
│
├── configs/
│   └── flux_lora_v1.toml          ← training config (increment version per run)
│
├── scripts/
│   ├── prepare_dataset.py         ← crop + resize raw photos to 1024x1024
│   ├── caption_dataset.py         ← writes .txt captions alongside training images
│   ├── upload_to_drive.py         ← syncs ComfyUI output → Google Drive via rclone
│   └── feedback.py                ← SQLite-based image rating + promotion tool
│
└── workflows/
    └── flux_lora_portrait.json    ← importable ComfyUI workflow
```

---

## Infrastructure

### RunPod Network Volume — `flux-lora-volume`

Persistent 200GB volume mounted at `/runpod-volume/` on every pod. **Never store anything important only on the pod disk** — it is destroyed when the pod stops.

```
/runpod-volume/
├── models/
│   ├── flux2-dev/              ← FLUX.2-dev weights (~60GB, downloaded once)
│   └── loras/                  ← trained LoRA .safetensors outputs
├── comfyui/                    ← ComfyUI install (installed once, persists)
│   ├── models/
│   │   ├── checkpoints/flux2-dev  ← symlink → /runpod-volume/models/flux2-dev
│   │   ├── loras/                 ← symlink → /runpod-volume/models/loras
│   │   ├── vae/                   ← ae.safetensors
│   │   └── clip/                  ← CLIP + T5 text encoders
│   ├── custom_nodes/
│   │   ├── ComfyUI-GGUF/
│   │   └── ComfyUI-Manager/
│   └── output/                    ← generated images land here
├── datasets/
│   └── myself/
│       ├── raw/                ← original uploaded photos (untouched)
│       └── train/              ← preprocessed 1024x1024 JPGs + .txt captions
├── outputs/                    ← training checkpoints + TensorBoard logs
├── configs/                    ← this git repo, cloned here on startup
├── feedback.db                 ← SQLite database for image ratings
└── rclone.conf                 ← Google Drive OAuth config (copied from laptop once)
```

### RunPod Pod

- **Template:** `flux-lora-trainer`
- **Docker image:** `YOUR_DOCKERHUB_USERNAME/flux-lora-pod:latest`
- **Recommended GPU:** A100 SXM 80GB at ~$1.49/hr (US-KS-2 data center)
- **Fallback GPU:** H100 80GB at ~$2.72/hr
- **Ports:** 8188 (ComfyUI), 6006 (TensorBoard), 22 (SSH)
- **SSH key:** `~/.ssh/runpod_key`

### Pod Startup Behaviour

The container start command (set in the RunPod template) runs automatically on every boot and does the following in order:

1. Creates all necessary directories on the network volume
2. Clones this git repo to `/runpod-volume/configs/` if not present, otherwise `git pull`
3. Copies `/runpod-volume/rclone.conf` to `/root/.config/rclone/rclone.conf` if it exists
4. Installs ComfyUI to `/runpod-volume/comfyui/` if not already installed (first boot only)
5. Installs ComfyUI-GGUF and ComfyUI-Manager custom nodes (first boot only)
6. Creates symlinks: LoRAs and FLUX.2 model into ComfyUI's model folders
7. Copies `workflows/flux_lora_portrait.json` into ComfyUI's workflow folder
8. Launches ComfyUI on port 8188
9. Keeps the container alive

### Training Framework

Kohya sd-scripts installed at `/opt/sd-scripts/` inside the Docker image. Training command:

```bash
cd /opt/sd-scripts
accelerate launch \
    --mixed_precision="bf16" \
    --num_processes=1 \
    flux_train_network.py \
    --config_file /runpod-volume/configs/configs/flux_lora_v1.toml
```

---

## Model Details

| Property | Value |
|----------|-------|
| Base model | FLUX.2 [dev] — 32B parameter flow matching transformer |
| Model path | `/runpod-volume/models/flux2-dev/` |
| HuggingFace repo | `black-forest-labs/FLUX.2-dev` |
| License | FLUX Non-Commercial License |
| LoRA framework | `networks.lora_flux` (kohya sd-scripts) |
| Training precision | bf16 |
| VRAM required | ~80GB (A100 SXM handles it natively, no quantization needed) |

---

## Training Config — `configs/flux_lora_v1.toml`

Key parameters and why they're set this way:

| Parameter | Value | Reason |
|-----------|-------|--------|
| `network_dim` | 32 | Higher rank than Flux.1 — captures facial detail in 32B model |
| `network_alpha` | 16 | Standard: half of network_dim |
| `learning_rate` | 1e-4 | Proven starting point for person LoRAs on FLUX.2 |
| `max_train_steps` | 1500 | FLUX.2 converges faster than Flux.1; 800–1500 is the sweet spot |
| `lr_scheduler` | cosine | Smooth decay, less likely to overfit than constant LR |
| `mixed_precision` | bf16 | Required for FLUX.2 training stability |
| `gradient_checkpointing` | true | Reduces VRAM usage during training |
| `num_repeats` | 10 | Each image seen 10x per epoch — compensates for small dataset |
| `keep_tokens` | 1 | Keeps trigger word at front of every caption |

**Versioning convention:** when changing training parameters, copy `flux_lora_v1.toml` to `flux_lora_v2.toml`, update `output_name` and `output_dir` to match, and log the run in `training_log.md`.

---

## Dataset

### Trigger Word

`ohwx man` — a nonsense string that won't conflict with existing model concepts. Used at the start of every training caption and in every generation prompt.

### Training Images

Located at `/runpod-volume/datasets/myself/train/`. Format: `0001.jpg`, `0002.jpg`, ... with matching `0001.txt`, `0002.txt`, ... caption files.

Target dataset size: 20–40 images. Quality matters more than quantity.

### Preprocessing Script — `scripts/prepare_dataset.py`

Takes raw photos from `/datasets/myself/raw/`, center-crops to square, resizes to 1024×1024, saves as JPG to `/datasets/myself/train/`. Run on the pod after uploading photos.

```bash
python /runpod-volume/configs/scripts/prepare_dataset.py \
    --src /runpod-volume/datasets/myself/raw \
    --dst /runpod-volume/datasets/myself/train
```

### Caption Script — `scripts/caption_dataset.py`

Writes a `.txt` caption file alongside each image. Cycles through varied template captions using the trigger word. Always manually review and edit captions after running — accurate captions are the biggest lever on LoRA quality.

```bash
python /runpod-volume/configs/scripts/caption_dataset.py \
    --dir /runpod-volume/datasets/myself/train \
    --trigger "ohwx man"
```

---

## ComfyUI

### Access

URL is shown in the pod's Connect tab next to port 8188. Format: `https://POD_ID-8188.proxy.runpod.net`

### Workflow

`workflows/flux_lora_portrait.json` is the base workflow. Load it in ComfyUI via the Load button (top-right). It is automatically copied to ComfyUI's workflow folder on pod startup.

Node graph: CheckpointLoaderSimple → LoraLoader → CLIPTextEncode (positive + negative) → EmptyLatentImage → KSampler → VAEDecode → SaveImage

### Sampler Settings for FLUX.2

| Setting | Value |
|---------|-------|
| Sampler | euler |
| Scheduler | simple |
| Steps | 28 (50 for maximum quality) |
| CFG | 4.0 |
| Denoise | 1.0 |
| Resolution | 1024×1024 |

Note: FLUX.2 uses real CFG (unlike Flux.1 which was guidance-distilled). Negative prompts work. Default negative: `blurry, low quality, deformed face, distorted hands, watermark, extra limbs, bad anatomy`

### LoRA Strength

Set in the LoraLoader node. Start at 0.8. Increase toward 1.0 if likeness is weak, decrease toward 0.65 if results are distorted.

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
Solo, different framing to Target 1. If Target 1 is full body → this is a tighter face shot, and vice versa.
```
ohwx man, sitting at a cafe, soft natural window light falling on face,
slight smile, relaxed confident posture, blurred background, close-up
portrait, wearing a clean fitted shirt, candid atmosphere, photorealistic,
film photography aesthetic
```

### Target 3 — Playing Guitar (Candid)
Eyes on strings, mid-action, not looking at camera.
```
ohwx man, playing acoustic guitar, candid moment, looking down at the
strings, absorbed in the music, soft warm indoor light, living room or
intimate venue setting, natural expression, shallow depth of field,
documentary photography style, photorealistic
```

### Target 4 — Reading a Book (Candid)
Genuinely reading, not posing. Eyes on the page.
```
ohwx man, reading a book at a café, soft window light, natural absorbed
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
Generate 4–8 seeds per prompt, keep the best. Face should be visible even in action shots.
- Jet skiing / water sports
- Skydiving
- Boat with friends (group shots fine here)
- Motorcycle
- Gym / working out
- Sharp outfit on a city street
- Travel / foreign city

---

## Feedback & Rating System — `scripts/feedback.py`

SQLite database at `/runpod-volume/feedback.db`. Stores image path, prompt, seed, LoRA version, rating (1–5), and notes. Images stay as files on disk — the database only stores metadata and paths.

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

The RunPod MCP server is installed globally in Claude Code. Use it to manage pods without leaving the editor.

```bash
# Verify it's connected
claude mcp list

# Example natural language commands inside Claude Code:
# "Start a RunPod pod using my flux-lora-trainer template with an A100 SXM"
# "Stop my RunPod pod"
# "List my running pods"
```

---

## Key Conventions

- **Never commit** model weights (`.safetensors`, `.ckpt`, `.bin`), training images, or generated outputs — all in `.gitignore`
- **Version configs** by copying `flux_lora_v1.toml` → `flux_lora_v2.toml` and updating `output_name` and `output_dir`
- **Log every run** in `training_log.md` with version, config, LR, steps, rank, and notes
- **Save workflows** back to `workflows/` after tweaking prompts in ComfyUI, then `git push`
- **All scripts run on the pod**, not locally — paths are hardcoded to `/runpod-volume/`
- **Git is the source of truth** for configs, scripts, and workflows — the pod is ephemeral

---

## Common Tasks for Claude Code

### Start a training run
```bash
# 1. SSH into running pod
ssh root@POD_IP -p PORT -i ~/.ssh/runpod_key

# 2. Pull latest configs
git -C /runpod-volume/configs pull

# 3. Run training inside tmux
tmux new -s training
cd /opt/sd-scripts
accelerate launch --mixed_precision=bf16 --num_processes=1 \
    flux_train_network.py \
    --config_file /runpod-volume/configs/configs/flux_lora_v1.toml
```

### Check training progress
```bash
# GPU usage
watch -n1 nvidia-smi

# TensorBoard (open port 6006 in browser)
tensorboard --logdir /runpod-volume/outputs/logs --host 0.0.0.0 --port 6006 &
```

### Upload photos for a new training run
```bash
scp -P PORT -i ~/.ssh/runpod_key ~/Photos/*.jpg \
    root@POD_IP:/runpod-volume/datasets/myself/raw/

python /runpod-volume/configs/scripts/prepare_dataset.py \
    --src /runpod-volume/datasets/myself/raw \
    --dst /runpod-volume/datasets/myself/train

python /runpod-volume/configs/scripts/caption_dataset.py \
    --dir /runpod-volume/datasets/myself/train \
    --trigger "ohwx man"
```

### End of session checklist
```bash
# 1. Rate generated images
python /runpod-volume/configs/scripts/feedback.py --session "v1_session"

# 2. Upload keepers to Google Drive
python /runpod-volume/configs/scripts/upload_to_drive.py --folder "v1_portraits"

# 3. Save workflow to git
cp /runpod-volume/comfyui/user/default/workflows/flux_lora_portrait.json \
   /runpod-volume/configs/workflows/
git -C /runpod-volume/configs add -A
git -C /runpod-volume/configs commit -m "Save session workflow and ratings"
git -C /runpod-volume/configs push

# 4. Stop the pod
```

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| ComfyUI not loading | `pkill -f "python main.py"` then `cd /runpod-volume/comfyui && python main.py --listen 0.0.0.0 --port 8188 &` |
| rclone upload failing | `rclone listremotes` — if `gdrive:` missing, rclone.conf not loaded. Check `/runpod-volume/rclone.conf` exists |
| OOM during training | Reduce `batch_size` to 1, ensure `gradient_checkpointing = true`, verify A100 80GB is the selected GPU |
| Generated hands look wrong | Add `detailed hands, correct hand anatomy` to positive prompt |
| LoRA not appearing in ComfyUI | Check symlink: `ls -la /runpod-volume/comfyui/models/loras/` |
| Git pull fails on pod | `git -C /runpod-volume/configs fetch --all && git -C /runpod-volume/configs reset --hard origin/main` |
| Training loss not decreasing | Check captions are accurate and varied, verify dataset has 20+ images, try lowering LR to `8e-5` |
