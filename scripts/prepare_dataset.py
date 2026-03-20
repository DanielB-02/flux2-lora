# scripts/prepare_dataset.py
#
# First-run preprocessing: center crop to square + resize to 1024x1024 + save as JPG.
#
# Usage:
#   python scripts/prepare_dataset.py \
#       --src /runpod-volume/datasets/myself/raw \
#       --dst /runpod-volume/datasets/myself/train

import argparse
from pathlib import Path
from PIL import Image

try:
    import pillow_heif
    pillow_heif.register_heif_opener()
except ImportError:
    pass


def parse_args():
    p = argparse.ArgumentParser(description="Resize photos for LoRA training")
    p.add_argument("--src",  required=True, help="Folder of raw photos")
    p.add_argument("--dst",  required=True, help="Output folder")
    p.add_argument("--size", type=int, default=1024, help="Output resolution (square). Default: 1024")
    return p.parse_args()


def center_crop_and_resize(img, size):
    """Center-crop to square, then resize to size x size."""
    w, h  = img.size
    side  = min(w, h)
    left  = (w - side) // 2
    top   = (h - side) // 2
    img   = img.crop((left, top, left + side, top + side))
    return img.resize((size, size), Image.LANCZOS)


def main():
    args = parse_args()
    src  = Path(args.src)
    dst  = Path(args.dst)
    dst.mkdir(parents=True, exist_ok=True)

    exts   = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".heic", ".heif"}
    images = sorted([f for f in src.iterdir() if f.suffix.lower() in exts])

    print(f"\nProcessing {len(images)} images -> {dst} ({args.size}x{args.size} JPG)\n")

    for i, fpath in enumerate(images, 1):
        img      = Image.open(fpath).convert("RGB")
        img      = center_crop_and_resize(img, args.size)
        out_path = dst / f"{i:04d}.jpg"
        img.save(str(out_path), "JPEG", quality=95)
        print(f"  [{i}/{len(images)}]  {fpath.name}  ->  {out_path.name}")

    print(f"\nDone. {len(images)} images saved to {dst}\n")


if __name__ == "__main__":
    main()