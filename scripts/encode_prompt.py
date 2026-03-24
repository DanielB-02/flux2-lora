"""Encode a prompt and save embeddings to disk. Runs as a separate process.
Uses GPU if available and free, otherwise CPU."""
import sys, torch
sys.stdout.reconfigure(line_buffering=True)
from diffusers import Flux2Pipeline
import subprocess

PROMPT = sys.argv[1]
MODEL_PATH = sys.argv[2] if len(sys.argv) > 2 else "/runpod-volume/models/flux2-dev"
NUM_IMAGES = int(sys.argv[3]) if len(sys.argv) > 3 else 1
OUT_PATH = "/tmp/prompt_embeds.pt"

# Check if GPU has enough free VRAM for the text encoder (~46GB)
use_gpu = False
if torch.cuda.is_available():
    r = subprocess.run(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
                       capture_output=True, text=True)
    free_mib = float(r.stdout.strip())
    use_gpu = free_mib > 48000  # need ~46GB for text encoder

device = "cuda" if use_gpu else "cpu"
print(f"Encoding on {device.upper()} ({NUM_IMAGES} variant{'s' if NUM_IMAGES > 1 else ''})...")

pipe = Flux2Pipeline.from_pretrained(MODEL_PATH, torch_dtype=torch.bfloat16)
if use_gpu:
    pipe.text_encoder.to("cuda")

embeds, mask = pipe.encode_prompt(
    prompt=PROMPT, device=device, num_images_per_prompt=NUM_IMAGES
)
torch.save({"prompt_embeds": embeds.cpu(), "prompt_attention_mask": mask.cpu()}, OUT_PATH)
print(f"Embeds: {embeds.shape} -> {OUT_PATH}")
