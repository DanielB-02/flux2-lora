# scripts/generate.py
#
# Generate images using FLUX.2-dev + trained LoRA.
# Runs on the pod with A100 80GB.
#
# Usage:
#   python /runpod-volume/configs/scripts/generate.py
#   python /runpod-volume/configs/scripts/generate.py --lora_strength 1.0 --steps 50
#   python /runpod-volume/configs/scripts/generate.py --prompt "ohwx man, custom prompt here"

import argparse
import torch
from pathlib import Path
from safetensors.torch import load_file
from diffusers import Flux2Pipeline


def apply_lora(pipe, lora_path, strength=0.8):
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
    p.add_argument("--prompt", default=None,
                    help="Single prompt to generate (overrides targets)")
    p.add_argument("--negative", default="blurry, low quality, deformed face, distorted hands, watermark, extra limbs, bad anatomy",
                    help="Negative prompt")
    p.add_argument("--steps", type=int, default=28, help="Sampling steps. Default: 28")
    p.add_argument("--cfg", type=float, default=4.0, help="CFG scale. Default: 4.0")
    p.add_argument("--seed", type=int, default=42, help="Random seed. Default: 42")
    p.add_argument("--width", type=int, default=1024)
    p.add_argument("--height", type=int, default=1024)
    p.add_argument("--targets_only", action="store_true",
                    help="Only generate the 5 target images")
    return p.parse_args()


TARGETS = {
    "target1_street_portrait": (
        "ohwx man, pay special attention that his beard is loaded correct, "
        "walking down a city street, wearing a well-fitted dark jacket and "
        "clean trousers, face clearly visible,  confident natural expression, soft "
        "golden hour sunlight, shallow depth of field, urban background, candid "
        "lifestyle photography, photorealistic, sharp focus"
    ),
    "target2_cafe_portrait": (
        "ohwx man, sitting at a cafe, soft natural window light falling on face, "
        "slight smile, relaxed confident posture, blurred background, close-up "
        "portrait, wearing a clean fitted shirt, candid atmosphere, photorealistic, "
        "film photography aesthetic"
    ),
    "target3_guitar": (
        "ohwx man, playing acoustic guitar, candid moment, looking down at the "
        "strings, absorbed in the music, soft warm indoor light, living room or "
        "intimate venue setting, natural expression, shallow depth of field, "
        "documentary photography style, photorealistic"
    ),
    "target4_reading": (
        "ohwx man, reading a book at a cafe, soft window light, natural absorbed "
        "expression, slight lean forward, coffee cup on table, background slightly "
        "blurred, candid street photography style, warm tones, photorealistic, "
        "sharp focus on face"
    ),
    "target5_nature": (
        "ohwx man, standing on a cliff overlooking dramatic mountain scenery, "
        "looking out at the view, casual outdoor clothing, backlit by golden hour "
        "light, candid moment, wide shot showing environment and figure, epic "
        "landscape photography, photorealistic, sharp"
    ),
    "target6_pullups": (
        "ohwx man, doing pull ups on an outdoor calisthenics bar, urban gym setting, "
        "mid-rep with arms fully engaged, athletic clothing, face and body clearly visible, "
        "natural effort expression, golden hour sunlight casting strong shadows, "
        "gritty urban background, candid action shot, documentary photography style, "
        "photorealistic, sharp focus"
    ),
}


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading FLUX.2-dev from {args.model_path}...")
    pipe = Flux2Pipeline.from_pretrained(args.model_path, dtype=torch.bfloat16)
    pipe.enable_model_cpu_offload()

    print(f"Applying LoRA from {args.lora_path} (strength={args.lora_strength})...")
    apply_lora(pipe, args.lora_path, args.lora_strength)

    generator = torch.Generator(device="cpu").manual_seed(args.seed)

    if args.prompt:
        prompts = {"custom": args.prompt}
    else:
        prompts = TARGETS

    for name, prompt in prompts.items():
        print(f"\nGenerating: {name}")
        print(f"  Prompt: {prompt[:80]}...")

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

        # Reset generator for next image with different seed
        generator = torch.Generator(device="cpu").manual_seed(args.seed + hash(name) % 10000)

    print(f"\nDone. {len(prompts)} images saved to {output_dir}")


if __name__ == "__main__":
    main()
