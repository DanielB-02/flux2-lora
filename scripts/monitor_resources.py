# scripts/monitor_resources.py
#
# Monitor system resources during image generation runs.
# Tracks GPU VRAM, GPU utilization, CPU, RAM, and disk usage.
#
# Usage:
#   python scripts/monitor_resources.py                    # continuous monitoring (2s interval)
#   python scripts/monitor_resources.py --interval 5       # custom interval
#   python scripts/monitor_resources.py --snapshot         # single snapshot and exit
#   python scripts/monitor_resources.py --log resources.csv # log to CSV file

import argparse
import csv
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import psutil


def get_gpu_stats():
    """Query nvidia-smi for GPU metrics. Returns dict or None if unavailable."""
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu,temperature.gpu,power.draw,power.limit",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode != 0:
            return None

        gpus = []
        for line in result.stdout.strip().split("\n"):
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 9:
                gpus.append({
                    "index": int(parts[0]),
                    "name": parts[1],
                    "vram_total_mib": float(parts[2]),
                    "vram_used_mib": float(parts[3]),
                    "vram_free_mib": float(parts[4]),
                    "gpu_util_pct": float(parts[5]),
                    "temp_c": float(parts[6]),
                    "power_w": float(parts[7]),
                    "power_limit_w": float(parts[8]),
                })
        return gpus
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None


def get_gpu_processes():
    """Get per-process GPU memory usage."""
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,used_gpu_memory,process_name",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode != 0:
            return []

        processes = []
        for line in result.stdout.strip().split("\n"):
            if not line.strip():
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 3:
                processes.append({
                    "pid": int(parts[0]),
                    "vram_mib": float(parts[1]),
                    "name": parts[2],
                })
        return processes
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return []


def get_system_stats():
    """Get CPU, RAM, and disk stats via psutil."""
    mem = psutil.virtual_memory()
    disk = psutil.disk_usage("/")

    return {
        "cpu_pct": psutil.cpu_percent(interval=0.1),
        "cpu_count": psutil.cpu_count(),
        "ram_total_gb": mem.total / (1024 ** 3),
        "ram_used_gb": mem.used / (1024 ** 3),
        "ram_available_gb": mem.available / (1024 ** 3),
        "ram_pct": mem.percent,
        "disk_total_gb": disk.total / (1024 ** 3),
        "disk_used_gb": disk.used / (1024 ** 3),
        "disk_free_gb": disk.free / (1024 ** 3),
        "disk_pct": disk.percent,
    }


def format_snapshot(gpu_stats, sys_stats, gpu_procs=None):
    """Format a human-readable snapshot of all resources."""
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = [f"\n{'=' * 60}", f"  Resource Snapshot — {ts}", f"{'=' * 60}"]

    # GPU section
    if gpu_stats:
        for gpu in gpu_stats:
            vram_used_pct = (gpu["vram_used_mib"] / gpu["vram_total_mib"]) * 100
            lines.append(f"\n  GPU {gpu['index']}: {gpu['name']}")
            lines.append(f"    VRAM:  {gpu['vram_used_mib']:,.0f} / {gpu['vram_total_mib']:,.0f} MiB "
                         f"({vram_used_pct:.1f}%) — {gpu['vram_free_mib']:,.0f} MiB free")
            lines.append(f"    Util:  {gpu['gpu_util_pct']:.0f}%")
            lines.append(f"    Temp:  {gpu['temp_c']:.0f}°C")
            lines.append(f"    Power: {gpu['power_w']:.0f}W / {gpu['power_limit_w']:.0f}W")
    else:
        lines.append("\n  GPU: not available")

    # GPU processes
    if gpu_procs:
        lines.append(f"\n  GPU Processes:")
        for proc in gpu_procs:
            lines.append(f"    PID {proc['pid']:>7}: {proc['vram_mib']:>8,.0f} MiB  {proc['name']}")

    # System section
    lines.append(f"\n  CPU:  {sys_stats['cpu_pct']:.1f}% across {sys_stats['cpu_count']} cores")
    lines.append(f"  RAM:  {sys_stats['ram_used_gb']:.1f} / {sys_stats['ram_total_gb']:.1f} GB "
                 f"({sys_stats['ram_pct']:.1f}%) — {sys_stats['ram_available_gb']:.1f} GB available")
    lines.append(f"  Disk: {sys_stats['disk_used_gb']:.1f} / {sys_stats['disk_total_gb']:.1f} GB "
                 f"({sys_stats['disk_pct']:.1f}%) — {sys_stats['disk_free_gb']:.1f} GB free")
    lines.append(f"{'=' * 60}")

    return "\n".join(lines)


def estimate_parallel_capacity(gpu_stats):
    """Estimate how many parallel image generations fit in VRAM.

    FLUX.2-dev in bfloat16 uses ~35-40GB VRAM for the model.
    Each 1024x1024 inference pass uses ~8-12GB additional VRAM.
    """
    if not gpu_stats:
        return None

    gpu = gpu_stats[0]
    vram_free = gpu["vram_free_mib"]
    vram_total = gpu["vram_total_mib"]
    vram_used = gpu["vram_used_mib"]

    # Measured constants for FLUX.2-dev on A100 80GB (with CPU offloading)
    # Model components: transformer 60GB + text_encoder 45GB + VAE 0.2GB = 105GB total
    # With enable_model_cpu_offload(), peak VRAM during 1-image gen: ~62,636 MiB
    PEAK_VRAM_1IMG_MIB = 62_636  # measured peak for 1 image at 1024x1024
    PER_EXTRA_IMAGE_MIB = 1_158  # measured: each additional batched image costs ~1.16 GB
    SAFETY_MARGIN_MIB = 2_000    # keep 2GB headroom to avoid OOM

    # If model is already loaded (CPU offload shows high VRAM during inference)
    if vram_used > 20_000:  # model actively running on GPU
        available_for_extra = vram_total - vram_used - SAFETY_MARGIN_MIB
        parallel_images = max(1, 1 + int(available_for_extra / PER_EXTRA_IMAGE_MIB))
        model_loaded = True
    else:
        # Model not yet running — estimate from measured baseline
        available_for_extra = vram_total - PEAK_VRAM_1IMG_MIB - SAFETY_MARGIN_MIB
        parallel_images = max(1, 1 + int(available_for_extra / PER_EXTRA_IMAGE_MIB))
        model_loaded = False

    return {
        "vram_total_mib": vram_total,
        "vram_used_mib": vram_used,
        "vram_free_mib": vram_free,
        "model_loaded": model_loaded,
        "peak_1img_vram_mib": PEAK_VRAM_1IMG_MIB,
        "per_extra_image_mib": PER_EXTRA_IMAGE_MIB,
        "available_for_extra_mib": (vram_total - vram_used - SAFETY_MARGIN_MIB) if model_loaded
            else (vram_total - PEAK_VRAM_1IMG_MIB - SAFETY_MARGIN_MIB),
        "estimated_parallel_images": parallel_images,
    }


def format_capacity_report(cap):
    """Format the parallel capacity estimate as a readable report."""
    lines = [
        f"\n{'=' * 60}",
        f"  Parallel Image Generation Capacity Estimate",
        f"  (based on measured FLUX.2-dev benchmarks)",
        f"{'=' * 60}",
        f"  VRAM Total:            {cap['vram_total_mib']:>8,.0f} MiB",
        f"  VRAM Used:             {cap['vram_used_mib']:>8,.0f} MiB",
        f"  VRAM Free:             {cap['vram_free_mib']:>8,.0f} MiB",
        f"  Model Active on GPU:   {'Yes' if cap['model_loaded'] else 'No'}",
        f"  Peak VRAM (1 img):     {cap['peak_1img_vram_mib']:>8,} MiB  (measured)",
        f"  Per Extra Image:       {cap['per_extra_image_mib']:>8,} MiB  (measured)",
        f"  Available for Extras:  {cap['available_for_extra_mib']:>8,.0f} MiB",
        f"  ─────────────────────────────────────────",
        f"  Estimated max batch (num_images_per_prompt): {cap['estimated_parallel_images']}",
        f"  Note: uses CPU offloading (model=105GB > 80GB VRAM)",
        f"  Batch is ~3.6x faster than sequential for 4 images",
        f"{'=' * 60}",
    ]
    return "\n".join(lines)


def write_csv_row(csv_path, gpu_stats, sys_stats, is_first):
    """Append a row to the CSV log file."""
    fieldnames = [
        "timestamp", "gpu_vram_used_mib", "gpu_vram_free_mib", "gpu_vram_total_mib",
        "gpu_util_pct", "gpu_temp_c", "gpu_power_w",
        "cpu_pct", "ram_used_gb", "ram_available_gb", "ram_pct",
    ]
    row = {"timestamp": datetime.now().isoformat()}

    if gpu_stats:
        gpu = gpu_stats[0]
        row.update({
            "gpu_vram_used_mib": gpu["vram_used_mib"],
            "gpu_vram_free_mib": gpu["vram_free_mib"],
            "gpu_vram_total_mib": gpu["vram_total_mib"],
            "gpu_util_pct": gpu["gpu_util_pct"],
            "gpu_temp_c": gpu["temp_c"],
            "gpu_power_w": gpu["power_w"],
        })

    row.update({
        "cpu_pct": sys_stats["cpu_pct"],
        "ram_used_gb": round(sys_stats["ram_used_gb"], 1),
        "ram_available_gb": round(sys_stats["ram_available_gb"], 1),
        "ram_pct": sys_stats["ram_pct"],
    })

    mode = "w" if is_first else "a"
    with open(csv_path, mode, newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if is_first:
            writer.writeheader()
        writer.writerow(row)


def parse_args():
    p = argparse.ArgumentParser(description="Monitor system resources during image generation")
    p.add_argument("--interval", type=float, default=2.0, help="Sampling interval in seconds (default: 2)")
    p.add_argument("--snapshot", action="store_true", help="Print a single snapshot and exit")
    p.add_argument("--capacity", action="store_true", help="Estimate parallel image generation capacity and exit")
    p.add_argument("--log", type=str, default=None, help="Path to CSV log file")
    return p.parse_args()


def main():
    args = parse_args()

    if args.snapshot or args.capacity:
        gpu_stats = get_gpu_stats()
        sys_stats = get_system_stats()
        gpu_procs = get_gpu_processes()
        print(format_snapshot(gpu_stats, sys_stats, gpu_procs))

        if args.capacity:
            cap = estimate_parallel_capacity(gpu_stats)
            if cap:
                print(format_capacity_report(cap))
            else:
                print("\n  Cannot estimate capacity — no GPU detected.")
        return

    # Continuous monitoring
    print("Monitoring resources... (Ctrl+C to stop)")
    is_first = True
    try:
        while True:
            gpu_stats = get_gpu_stats()
            sys_stats = get_system_stats()
            gpu_procs = get_gpu_processes()
            print(format_snapshot(gpu_stats, sys_stats, gpu_procs))

            if args.log:
                write_csv_row(args.log, gpu_stats, sys_stats, is_first)
                is_first = False

            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\nMonitoring stopped.")


if __name__ == "__main__":
    main()
