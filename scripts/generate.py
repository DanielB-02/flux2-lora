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
import os
import torch
from pathlib import Path

try:
    from diffusers import Flux2Pipeline as Pipeline
except ImportError:
    from diffusers import FluxPipeline as Pipeline


def parse_args():
    p = argparse.ArgumentParser(description="Generate images with FLUX.2 + LoRA")
    p.add_argument("--model_path", default="/runpod-volume/models/flux2-dev",
                    help="Path to FLUX.2-dev model directory")
    p.add_argument("--lora_path", default="/runpod-volume/outputs/lora_v1/flux2_lora_v1.safetensors",
                    help="Path to LoRA safetensors file")
    p.add_argument("--lora_strength", type=float, default=0.8,
                    help="LoRA strength (0.6-1.2). Default: 0.8")
    p.add_argument("--output_dir", default="/runpod-volume/comfyui/output",
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
        "ohwx man, walking down a city street, wearing a well-fitted dark jacket and "
        "clean trousers, face clearly visible, confident natural expression, soft "
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
}


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading FLUX.2-dev from {args.model_path}...")
    pipe = Pipeline.from_pretrained(args.model_path, torch_dtype=torch.bfloat16)
    pipe.enable_model_cpu_offload()

    print(f"Loading LoRA from {args.lora_path} (strength: {args.lora_strength})...")
    from diffusers.loaders.lora_conversion_utils import _convert_kohya_flux_lora_to_diffusers
    from safetensors.torch import load_file
    kohya_state_dict = load_file(args.lora_path)
    try:
        converted = _convert_kohya_flux_lora_to_diffusers(kohya_state_dict)
        pipe.load_lora_weights(converted, adapter_name="lora")
        pipe.set_adapters(["lora"], adapter_weights=[args.lora_strength])
        print(f"  LoRA loaded and converted from kohya format")
    except Exception as e:
        print(f"  Warning: Could not convert LoRA ({e}), trying direct load...")
        pipe.load_lora_weights(args.lora_path)
        pipe.set_adapters(adapter_weights=[args.lora_strength])

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
