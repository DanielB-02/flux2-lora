# scripts/generate.py
#
# Generate images using FLUX.2-dev + trained LoRA.
# Runs on the pod with A100 80GB.
#
# Uses encode-once approach: text encoder runs in a subprocess to encode
# prompts, then the transformer + VAE stay on GPU permanently (~62GB).
# This gives 6.8x speedup over CPU offloading (40s vs 275s per image).
#
# Features:
#   - Interactive mode with live LoRA strength adjustment
#   - Parallel batch generation via num_images_per_prompt
#   - JSONL metadata log on pod (synced to local TinyDB via sync_generations.py)
#   - Encode-once: text encoder loads temporarily, transformer stays on GPU
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
#   variants <n>            — change number of parallel variants (1-14)
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

ENCODE_SCRIPT = Path(__file__).parent / "encode_prompt.py"
GDRIVE_DEST = "gdrive:FluxLoRA/generated"


def flush_stdin():
    """Discard any buffered stdin lines (e.g. from multi-line paste)."""
    while select.select([sys.stdin], [], [], 0.0)[0]:
        sys.stdin.readline()


def encode_prompt_subprocess(prompt, model_path, num_images_per_prompt=1):
    """Encode a prompt in a separate process so GPU memory is fully freed after.

    Returns (prompt_embeds, prompt_attention_mask) tensors on CPU.
    """
    result = subprocess.run(
        [sys.executable, str(ENCODE_SCRIPT), prompt, model_path, str(num_images_per_prompt)],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        print(f"  Encoding failed: {result.stderr.strip()}")
        return None, None
    for line in result.stdout.strip().split("\n"):
        print(f"  {line}")

    cached = torch.load("/tmp/prompt_embeds.pt", weights_only=True)
    return cached["prompt_embeds"], cached.get("prompt_attention_mask")


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

    for prefix in sorted(prefixes):
        down = lora_sd[f"{prefix}.lora_down.weight"]
        up = lora_sd[f"{prefix}.lora_up.weight"]
        alpha = lora_sd[f"{prefix}.alpha"].item()
        dim = down.shape[0]
        unit_scale = alpha / dim
        unit_delta = (up.float() @ down.float()) * unit_scale

        def apply_delta(weight, delta, s=strength):
            scaled = (delta * s).to(device=weight.device, dtype=weight.dtype)
            weight.data += scaled

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
        "ohwx man, with his neat natural beard, clean hair on the sides, tall and lean with slight muscle build, "
        "natural arm hair, playing electric guitar, gazing into the crowd with a confident subtle smile, "
        "wearing a fitted burgundy henley with arm sleeves rolled up and dark jeans, no rings, "
        "no jewelry, moody warm lighting, amplifiers in background, shallow depth of field, "
        "concert photography style, photorealistic"
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
    p.add_argument("--lora_strength", type=float, default=0.9,
                    help="LoRA strength (0.6-1.2). Default: 0.9")
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

    prompt_cache = {}  # prompt_text -> {num_variants -> (prompt_embeds, prompt_attention_mask)}

    # Load pipeline WITHOUT text encoder, put on GPU
    print(f"\nLoading FLUX.2-dev (transformer + VAE only) from {args.model_path}...")
    pipe = Flux2Pipeline.from_pretrained(
        args.model_path, text_encoder=None, tokenizer=None, torch_dtype=torch.bfloat16
    )
    pipe.to("cuda")

    print(f"Applying LoRA from {args.lora_path} (strength={args.lora_strength})...")
    apply_lora(pipe, args.lora_path, args.lora_strength)
    num_variants = args.num_variants

    gen_count = 0

    print(f"\nReady. Type a target name, 'all', or a custom prompt. 'quit' to exit.")
    print(f"Commands: seed <value>, variants <n>")
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

        # Check for near-miss command typos
        first_word = user_input.lower().split()[0] if user_input.split() else ""
        commands = ["seed", "variants", "quit", "exit", "q", "all"]
        if first_word not in commands and first_word not in TARGETS and not user_input.startswith("ohwx"):
            # Likely a typo or unknown command — confirm before treating as a prompt
            confirm = input(f"  Generate image with prompt \"{user_input[:60]}...\"? (y/n): ").strip().lower()
            if confirm != "y":
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

            # Encode prompt on demand, cache by (prompt, num_variants)
            cache_key = (prompt, num_variants)
            if cache_key not in prompt_cache:
                print(f"  Encoding prompt ({num_variants} variant{'s' if num_variants > 1 else ''})...")
                embeds, mask = encode_prompt_subprocess(prompt, args.model_path, num_variants)
                if embeds is None:
                    print("  Skipping — encoding failed.")
                    continue
                prompt_cache[cache_key] = (embeds, mask)

            prompt_embeds, prompt_mask = prompt_cache[cache_key]

            print(f"\nGenerating: {name} ({num_variants} variant{'s' if num_variants > 1 else ''} in parallel)")
            print(f"  Prompt: {prompt[:80]}...")
            print(f"  Base seed: {base_seed}")

            generator = torch.Generator(device="cpu").manual_seed(base_seed)

            try:
                t0 = time.time()
                with torch.inference_mode():
                    images = pipe(
                        prompt_embeds=prompt_embeds.to("cuda"),
                        num_inference_steps=args.steps,
                        guidance_scale=args.cfg,
                        height=args.height,
                        width=args.width,
                        generator=generator,
                        num_images_per_prompt=1,  # already batched via prompt_embeds
                    ).images
                gen_time = time.time() - t0
            except KeyboardInterrupt:
                print("\n  Generation cancelled.")
                gc.collect()
                torch.cuda.empty_cache()
                flush_stdin()
                break

            per_image_time = round(gen_time / len(images), 1)
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
                    "lora_strength": args.lora_strength,
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
