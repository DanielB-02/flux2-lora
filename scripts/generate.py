# scripts/generate.py
#
# Generate images using FLUX.2-dev + trained LoRA.
# Runs on the pod with A100 80GB. Uses CPU offloading since the full model
# (105GB) exceeds VRAM (80GB).
#
# Features:
#   - Interactive mode with live LoRA strength adjustment
#   - Parallel batch generation via num_images_per_prompt (3.6x faster than sequential)
#   - JSONL metadata log on pod (synced to local TinyDB via sync_generations.py)
#
# Usage:
#   python /runpod-volume/configs/scripts/generate.py                # interactive mode
#   python /runpod-volume/configs/scripts/generate.py --list_targets # show available targets
#   python /runpod-volume/configs/scripts/generate.py --num_variants 6
#
# Interactive commands:
#   target3_guitar          — generate a specific target (× num_variants in parallel)
#   all                     — generate all targets
#   ohwx man, custom prompt — generate from a custom prompt
#   variants <n>            — change number of parallel variants
#   strength <value>        — adjust LoRA strength (no model reload)
#   seed <value>            — change base seed
#   quit                    — exit

import argparse
import gc
import json
import select
import subprocess
import sys
import time
import torch
from datetime import datetime, timezone
from pathlib import Path
from safetensors.torch import load_file
from diffusers import Flux2Pipeline


def flush_stdin():
    """Discard any buffered stdin lines (e.g. from multi-line paste)."""
    while select.select([sys.stdin], [], [], 0.0)[0]:
        sys.stdin.readline()

GDRIVE_DEST = "gdrive:FluxLoRA/generated"


def apply_lora(pipe, lora_path, strength=1):
    """Manually merge kohya-format LoRA weights into the Flux2 transformer.

    Returns a list of (weight_tensor, unit_delta) tuples so strength can be
    adjusted later without reloading the model.
    """
    lora_sd = load_file(lora_path)
    transformer = pipe.transformer

    prefixes = set()
    for k in lora_sd:
        base = k.rsplit(".", 2)[0] if ".lora_" in k else k.rsplit(".", 1)[0]
        prefixes.add(base)

    applied = 0
    deltas = []  # [(weight_tensor, unit_delta)]

    for prefix in sorted(prefixes):
        down = lora_sd[f"{prefix}.lora_down.weight"]
        up = lora_sd[f"{prefix}.lora_up.weight"]
        alpha = lora_sd[f"{prefix}.alpha"].item()
        dim = down.shape[0]
        unit_scale = alpha / dim
        unit_delta = (up.float() @ down.float()) * unit_scale

        def apply_delta(weight, delta, s=strength):
            scaled = (delta * s).to(weight.dtype)
            weight.data += scaled
            deltas.append((weight, delta))

        if prefix.startswith("lora_unet_double_blocks_"):
            parts = prefix.replace("lora_unet_double_blocks_", "").split("_", 1)
            block_idx = int(parts[0])
            layer_name = parts[1]
            block = transformer.transformer_blocks[block_idx]

            if layer_name == "img_attn_qkv":
                q_d, k_d, v_d = unit_delta.chunk(3, dim=0)
                apply_delta(block.attn.to_q.weight, q_d)
                apply_delta(block.attn.to_k.weight, k_d)
                apply_delta(block.attn.to_v.weight, v_d)
            elif layer_name == "img_attn_proj":
                apply_delta(block.attn.to_out[0].weight, unit_delta)
            elif layer_name == "img_mlp_0":
                apply_delta(block.ff.linear_in.weight, unit_delta)
            elif layer_name == "img_mlp_2":
                apply_delta(block.ff.linear_out.weight, unit_delta)
            elif layer_name == "txt_attn_qkv":
                q_d, k_d, v_d = unit_delta.chunk(3, dim=0)
                apply_delta(block.attn.add_q_proj.weight, q_d)
                apply_delta(block.attn.add_k_proj.weight, k_d)
                apply_delta(block.attn.add_v_proj.weight, v_d)
            elif layer_name == "txt_attn_proj":
                apply_delta(block.attn.to_add_out.weight, unit_delta)
            elif layer_name == "txt_mlp_0":
                apply_delta(block.ff_context.linear_in.weight, unit_delta)
            elif layer_name == "txt_mlp_2":
                apply_delta(block.ff_context.linear_out.weight, unit_delta)
            else:
                continue

        elif prefix.startswith("lora_unet_single_blocks_"):
            parts = prefix.replace("lora_unet_single_blocks_", "").split("_", 1)
            block_idx = int(parts[0])
            layer_name = parts[1]
            block = transformer.single_transformer_blocks[block_idx]

            if layer_name == "linear1":
                apply_delta(block.attn.to_qkv_mlp_proj.weight, unit_delta)
            elif layer_name == "linear2":
                apply_delta(block.attn.to_out.weight, unit_delta)
            else:
                continue
        else:
            continue

        applied += 1

    print(f"  Applied {applied}/{len(prefixes)} LoRA layers (strength={strength})")
    return deltas


def update_lora_strength(deltas, old_strength, new_strength):
    """Adjust LoRA strength without reloading the model."""
    diff = new_strength - old_strength
    for weight, unit_delta in deltas:
        weight.data += (unit_delta * diff).to(weight.dtype)
    print(f"  LoRA strength: {old_strength} -> {new_strength}")


TARGETS = {
    "target1_street_portrait": (
        "ohwx man, with his full medium-length natural beard, "
        "walking along a city sidewalk, noticeably taller than nearby pedestrians, "
        "wearing a well-fitted dark jacket and clean trousers, face clearly visible,  "
        "confident natural expression, soft golden hour sunlight, "
        "shallow depth of field, urban background, candid lifestyle photography, "
        "photorealistic, sharp focus"
    ),
    "target2_cafe_portrait": (
        "ohwx man, with his full medium-length natural beard, sitting at a cafe, soft natural window light falling on face, "
        "slight smile, relaxed confident posture, blurred background, close-up "
        "portrait, wearing a clean fitted shirt, candid atmosphere, photorealistic, "
        "film photography aesthetic"
    ),
    "target3_guitar": (
        "ohwx man, with his full medium-length natural beard, playing electric guitar on a small stage, "
        "moody dim lighting with colored stage lights, candid moment, looking down at the "
        "fretboard, absorbed in the music, intimate bar or club venue, natural expression, "
        "shallow depth of field, concert photography style, photorealistic"
    ),
    "target4_reading": (
        "ohwx man, with his full medium-length natural beard, reading at a cafe, soft window light, natural absorbed "
        "expression, sitting comfortably, holding a single open book in both hands, "
        "coffee cup on table, background slightly blurred, "
        "candid street photography style, warm tones, photorealistic, "
        "sharp focus on face"
    ),
    "target5_nature": (
        "ohwx man, with his full medium-length natural beard, "
        "standing on a hillside with the Swiss Alps visible in the distance, green rolling landscape, "
        "looking out at the view, casual outdoor clothing, backlit by golden hour "
        "light, candid moment, wide shot showing environment and figure, epic "
        "landscape photography, photorealistic, sharp"
    ),
    "target6_pullups": (
        "ohwx man, with his full medium-length natural beard, doing pull ups on an outdoor calisthenics bar, urban gym setting, "
        "mid-rep with arms fully engaged, athletic clothing, face and body clearly visible, "
        "natural effort expression, golden hour sunlight casting strong shadows, "
        "gritty urban background, candid action shot, documentary photography style, "
        "photorealistic, sharp focus"
    ),
}


def get_gdrive_link(filepath):
    """Get shareable Google Drive link for an uploaded file."""
    gdrive_path = f"{GDRIVE_DEST}/{Path(filepath).name}"
    result = subprocess.run(
        ["rclone", "link", gdrive_path],
        capture_output=True, text=True, timeout=30,
    )
    if result.returncode == 0:
        return result.stdout.strip()
    return None


def append_jsonl(log_path, record):
    """Append a single JSON record to the JSONL log file."""
    with open(log_path, "a") as f:
        f.write(json.dumps(record) + "\n")


def save_upload_and_log(image, out_path, log_path, record):
    """Save image, upload to Google Drive, fetch link, and append to JSONL log."""
    image.save(str(out_path))
    print(f"    Saved: {out_path}")

    # Upload to Google Drive
    result = subprocess.run(
        ["rclone", "copy", str(out_path), GDRIVE_DEST],
        capture_output=True, text=True,
    )
    if result.returncode == 0:
        print(f"    Uploaded to {GDRIVE_DEST}/{out_path.name}")
        try:
            gdrive_url = get_gdrive_link(out_path)
            if gdrive_url:
                record["gdrive_url"] = gdrive_url
                print(f"    Drive link: {gdrive_url}")
        except Exception:
            pass
    else:
        print(f"    Upload failed: {result.stderr.strip()}")

    # Append to JSONL log
    append_jsonl(log_path, record)


def parse_args():
    p = argparse.ArgumentParser(description="Generate images with FLUX.2 + LoRA")
    p.add_argument("--model_path", default="/runpod-volume/models/flux2-dev",
                    help="Path to FLUX.2-dev model directory")
    p.add_argument("--lora_path", default="/runpod-volume/outputs/lora_v1/flux2_lora_v1.safetensors",
                    help="Path to LoRA safetensors file")
    p.add_argument("--lora_strength", type=float, default=0.8,
                    help="LoRA strength (0.6-1.2). Default: 0.8")
    p.add_argument("--output_dir", default="/runpod-volume/generated",
                    help="Output directory for generated images")
    p.add_argument("--list_targets", action="store_true",
                    help="List available target names and exit")
    p.add_argument("--negative", default="blurry, low quality, deformed face, distorted hands, watermark, extra limbs, bad anatomy",
                    help="Negative prompt")
    p.add_argument("--steps", type=int, default=28, help="Sampling steps. Default: 28")
    p.add_argument("--cfg", type=float, default=4.0, help="CFG scale. Default: 4.0")
    p.add_argument("--seed", type=int, default=42, help="Random seed. Default: 42")
    p.add_argument("--width", type=int, default=1024)
    p.add_argument("--height", type=int, default=1024)
    p.add_argument("--num_variants", type=int, default=4,
                    help="Number of parallel seed variants per prompt. Default: 4")
    p.add_argument("--log_path", default="/runpod-volume/generations.jsonl",
                    help="Path to JSONL log file for generation metadata")
    return p.parse_args()


def main():
    args = parse_args()

    if args.list_targets:
        print("Available targets:")
        for name in TARGETS:
            print(f"  {name}")
        sys.exit(0)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    log_path = args.log_path
    existing = 0
    if Path(log_path).exists():
        existing = sum(1 for _ in open(log_path))
    print(f"Generation log: {log_path} ({existing} existing records)")

    print(f"Loading FLUX.2-dev from {args.model_path}...")
    pipe = Flux2Pipeline.from_pretrained(args.model_path, torch_dtype=torch.bfloat16)
    print("Enabling CPU offloading (model=105GB > 80GB VRAM)...")
    pipe.enable_model_cpu_offload()

    print(f"Applying LoRA from {args.lora_path} (strength={args.lora_strength})...")
    lora_deltas = apply_lora(pipe, args.lora_path, args.lora_strength)
    current_strength = args.lora_strength
    num_variants = args.num_variants

    gen_count = 0

    print(f"\nReady. Type a target name, 'all', or a custom prompt. 'quit' to exit.")
    print(f"Commands: strength <value>, seed <value>, variants <n>")
    print(f"Generating {num_variants} parallel variant(s) per prompt.")
    print(f"Available targets: {', '.join(TARGETS.keys())}\n")

    while True:
        try:
            user_input = input(">>> ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "q"):
            break

        # Command: strength <value>
        if user_input.lower().startswith("strength "):
            try:
                new_strength = float(user_input.split()[1])
                update_lora_strength(lora_deltas, current_strength, new_strength)
                current_strength = new_strength
            except (ValueError, IndexError):
                print("Usage: strength <value> (e.g., strength 0.8)")
            continue

        # Command: seed <value>
        if user_input.lower().startswith("seed "):
            try:
                args.seed = int(user_input.split()[1])
                print(f"  Seed set to {args.seed}")
            except (ValueError, IndexError):
                print("Usage: seed <value> (e.g., seed 123)")
            continue

        # Command: variants <n>
        if user_input.lower().startswith("variants "):
            try:
                num_variants = max(1, int(user_input.split()[1]))
                print(f"  Variants set to {num_variants}")
            except (ValueError, IndexError):
                print("Usage: variants <n> (e.g., variants 4)")
            continue

        # Determine which prompts to generate
        if user_input.lower() == "all":
            prompts = TARGETS
        elif user_input in TARGETS:
            prompts = {user_input: TARGETS[user_input]}
        else:
            prompts = {"custom": user_input}

        for name, prompt in prompts.items():
            base_seed = args.seed + hash(name) % 10000

            print(f"\nGenerating: {name} ({num_variants} variant{'s' if num_variants > 1 else ''} in parallel)")
            print(f"  Prompt: {prompt[:80]}...")
            print(f"  Base seed: {base_seed}")

            generator = torch.Generator(device="cpu").manual_seed(base_seed)

            try:
                t0 = time.time()
                with torch.inference_mode():
                    images = pipe(
                        prompt=prompt,
                        num_inference_steps=args.steps,
                        guidance_scale=args.cfg,
                        height=args.height,
                        width=args.width,
                        generator=generator,
                        num_images_per_prompt=num_variants,
                    ).images
                gen_time = time.time() - t0
            except KeyboardInterrupt:
                print("\n  Generation cancelled.")
                gc.collect()
                torch.cuda.empty_cache()
                flush_stdin()
                break

            per_image_time = round(gen_time / num_variants, 1)
            print(f"  Generated {len(images)} images in {gen_time:.1f}s ({per_image_time}s/img)")

            # Save, upload, and log each image
            for variant_idx, image in enumerate(images):
                gen_count += 1
                out_path = output_dir / f"{name}_seed{base_seed}_v{variant_idx}_{gen_count:03d}.png"

                record = {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "prompt": prompt,
                    "target_name": name,
                    "seed": base_seed,
                    "variant_index": variant_idx,
                    "lora_strength": current_strength,
                    "lora_path": args.lora_path,
                    "steps": args.steps,
                    "cfg": args.cfg,
                    "width": args.width,
                    "height": args.height,
                    "filename": out_path.name,
                    "output_dir": str(output_dir),
                    "gdrive_url": None,
                    "generation_time_s": round(gen_time, 1),
                    "batch_size": num_variants,
                    "model_path": args.model_path,
                }

                save_upload_and_log(image, out_path, log_path, record)

            del images
            gc.collect()
            torch.cuda.empty_cache()

        flush_stdin()

    del pipe
    gc.collect()
    torch.cuda.empty_cache()
    print("Done.")


if __name__ == "__main__":
    main()
