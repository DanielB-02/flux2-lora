# scripts/generate.py
#
# Generate images using FLUX.2-dev + trained LoRA.
# Runs on the pod with A100 80GB.
#
# Usage:
#   python /runpod-volume/configs/scripts/generate.py                # interactive mode
#   python /runpod-volume/configs/scripts/generate.py --list_targets # show available targets
#
# Interactive commands:
#   target3_guitar          — generate a specific target
#   all                     — generate all targets
#   ohwx man, custom prompt — generate from a custom prompt
#   quit                    — exit

import argparse
import gc
import subprocess
import sys
import torch
from pathlib import Path
from safetensors.torch import load_file
from diffusers import Flux2Pipeline

GDRIVE_DEST = "gdrive:FluxLoRA/generated"


def apply_lora(pipe, lora_path, strength=1):
    """Manually merge kohya-format LoRA weights into the Flux2 transformer."""
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
        scale = strength * (alpha / dim)
        delta = (up.float() @ down.float()) * scale

        if prefix.startswith("lora_unet_double_blocks_"):
            parts = prefix.replace("lora_unet_double_blocks_", "").split("_", 1)
            block_idx = int(parts[0])
            layer_name = parts[1]
            block = transformer.transformer_blocks[block_idx]

            if layer_name == "img_attn_qkv":
                q_d, k_d, v_d = delta.chunk(3, dim=0)
                block.attn.to_q.weight.data += q_d.to(block.attn.to_q.weight.dtype)
                block.attn.to_k.weight.data += k_d.to(block.attn.to_k.weight.dtype)
                block.attn.to_v.weight.data += v_d.to(block.attn.to_v.weight.dtype)
            elif layer_name == "img_attn_proj":
                block.attn.to_out[0].weight.data += delta.to(block.attn.to_out[0].weight.dtype)
            elif layer_name == "img_mlp_0":
                block.ff.linear_in.weight.data += delta.to(block.ff.linear_in.weight.dtype)
            elif layer_name == "img_mlp_2":
                block.ff.linear_out.weight.data += delta.to(block.ff.linear_out.weight.dtype)
            elif layer_name == "txt_attn_qkv":
                q_d, k_d, v_d = delta.chunk(3, dim=0)
                block.attn.add_q_proj.weight.data += q_d.to(block.attn.add_q_proj.weight.dtype)
                block.attn.add_k_proj.weight.data += k_d.to(block.attn.add_k_proj.weight.dtype)
                block.attn.add_v_proj.weight.data += v_d.to(block.attn.add_v_proj.weight.dtype)
            elif layer_name == "txt_attn_proj":
                block.attn.to_add_out.weight.data += delta.to(block.attn.to_add_out.weight.dtype)
            elif layer_name == "txt_mlp_0":
                block.ff_context.linear_in.weight.data += delta.to(block.ff_context.linear_in.weight.dtype)
            elif layer_name == "txt_mlp_2":
                block.ff_context.linear_out.weight.data += delta.to(block.ff_context.linear_out.weight.dtype)
            else:
                continue

        elif prefix.startswith("lora_unet_single_blocks_"):
            parts = prefix.replace("lora_unet_single_blocks_", "").split("_", 1)
            block_idx = int(parts[0])
            layer_name = parts[1]
            block = transformer.single_transformer_blocks[block_idx]

            if layer_name == "linear1":
                block.attn.to_qkv_mlp_proj.weight.data += delta.to(block.attn.to_qkv_mlp_proj.weight.dtype)
            elif layer_name == "linear2":
                block.attn.to_out.weight.data += delta.to(block.attn.to_out.weight.dtype)
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

    print(f"Loading FLUX.2-dev from {args.model_path}...")
    pipe = Flux2Pipeline.from_pretrained(args.model_path, dtype=torch.bfloat16)
    pipe.enable_model_cpu_offload()

    print(f"Applying LoRA from {args.lora_path} (strength={args.lora_strength})...")
    apply_lora(pipe, args.lora_path, args.lora_strength)

    print("\nReady. Type a target name, 'all', or a custom prompt. 'quit' to exit.")
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

        if user_input.lower() == "all":
            prompts = TARGETS
        elif user_input in TARGETS:
            prompts = {user_input: TARGETS[user_input]}
        else:
            prompts = {"custom": user_input}

        for name, prompt in prompts.items():
            print(f"\nGenerating: {name}")
            print(f"  Prompt: {prompt[:80]}...")

            generator = torch.Generator(device="cpu").manual_seed(
                args.seed + hash(name) % 10000
            )

            with torch.inference_mode():
                image = pipe(
                    prompt=prompt,
                    num_inference_steps=args.steps,
                    guidance_scale=args.cfg,
                    height=args.height,
                    width=args.width,
                    generator=generator,
                ).images[0]

            out_path = output_dir / f"{name}_seed{args.seed}.png"
            image.save(str(out_path))
            print(f"  Saved: {out_path}")

            result = subprocess.run(
                ["rclone", "copy", str(out_path), GDRIVE_DEST],
                capture_output=True, text=True,
            )
            if result.returncode == 0:
                print(f"  Uploaded to {GDRIVE_DEST}/{out_path.name}")
            else:
                print(f"  Upload failed: {result.stderr.strip()}")

            del image
            gc.collect()
            torch.cuda.empty_cache()

    del pipe
    gc.collect()
    torch.cuda.empty_cache()
    print("Done.")


if __name__ == "__main__":
    main()
