# scripts/caption_dataset.py
#
# Writes caption files for every image in a training folder.
# Each caption = trigger word + varied descriptor.
#
# Usage:
#   python scripts/caption_dataset.py \
#       --dir /runpod-volume/datasets/myself/train \
#       --trigger "ohwx man"

import os
import argparse
from pathlib import Path

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dir",     required=True, help="Training images folder")
    p.add_argument("--trigger", required=True, help="Your trigger word, e.g. 'ohwx man'")
    return p.parse_args()

def main():
    args    = parse_args()
    d       = Path(args.dir)
    trigger = args.trigger

    # Varied caption templates — diversity helps the model generalise
    templates = [
        f"{trigger}, portrait photo, natural daylight, looking at camera",
        f"{trigger}, close-up face, soft studio lighting, sharp focus",
        f"{trigger}, side profile, indoor lighting, neutral expression",
        f"{trigger}, candid photo, outdoor daylight, casual clothing",
        f"{trigger}, upper body shot, warm lighting, slight smile",
        f"{trigger}, portrait, overcast light, relaxed expression",
        f"{trigger}, headshot, golden hour light, looking slightly off-camera",
        f"{trigger}, face portrait, diffused window light, calm expression",
    ]

    images = sorted([f for f in d.iterdir() if f.suffix.lower() in {".jpg",".jpeg",".png"}])

    print(f"Writing captions for {len(images)} images in {d}")
    for i, img in enumerate(images):
        caption = templates[i % len(templates)]
        txt     = img.with_suffix(".txt")
        txt.write_text(caption)
        print(f"  {img.name}  →  {caption}")

    print(f"\nDone. {len(images)} caption files written.")
    print("Review them manually and edit any that don't match the actual photo.")

if __name__ == "__main__":
    main()