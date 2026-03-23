# scripts/sync_generations.py
#
# Pulls the JSONL generation log from the pod and merges it into the local
# TinyDB at generations.json in the project root.
#
# Usage:
#   python scripts/sync_generations.py                          # uses default pod connection
#   python scripts/sync_generations.py --host 1.2.3.4 --port 22
#   python scripts/sync_generations.py --query                  # show all local records
#   python scripts/sync_generations.py --query --seed 42        # filter by seed
#   python scripts/sync_generations.py --query --target target1_street_portrait

import argparse
import json
import subprocess
import sys
from pathlib import Path
from tinydb import TinyDB, where


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = PROJECT_ROOT / "generations.json"
REMOTE_LOG = "/runpod-volume/generations.jsonl"


def sync_from_pod(host, port, key_path):
    """SCP the JSONL log from the pod and merge new records into local TinyDB."""
    local_jsonl = PROJECT_ROOT / "generations_remote.jsonl"

    print(f"Fetching {REMOTE_LOG} from {host}:{port}...")
    result = subprocess.run(
        [
            "scp", "-o", "StrictHostKeyChecking=no",
            "-P", str(port), "-i", key_path,
            f"root@{host}:{REMOTE_LOG}", str(local_jsonl),
        ],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        print(f"SCP failed: {result.stderr.strip()}")
        sys.exit(1)

    # Read remote records
    remote_records = []
    with open(local_jsonl) as f:
        for line in f:
            line = line.strip()
            if line:
                remote_records.append(json.loads(line))

    print(f"Remote log: {len(remote_records)} records")

    # Merge into local TinyDB (deduplicate by timestamp + filename)
    db = TinyDB(str(DB_PATH))
    existing = {(r["timestamp"], r["filename"]) for r in db.all()}

    new_count = 0
    for record in remote_records:
        key = (record.get("timestamp"), record.get("filename"))
        if key not in existing:
            db.insert(record)
            existing.add(key)
            new_count += 1

    db.close()
    local_jsonl.unlink()  # clean up temp file

    print(f"Local DB: {DB_PATH}")
    print(f"  {new_count} new records added, {len(existing)} total")


def query_db(target=None, seed=None):
    """Query and display local generation records."""
    if not DB_PATH.exists():
        print(f"No local database found at {DB_PATH}")
        print("Run sync first: python scripts/sync_generations.py")
        sys.exit(1)

    db = TinyDB(str(DB_PATH))
    records = db.all()

    if target:
        records = db.search(where("target_name") == target)
    if seed is not None:
        records = [r for r in records if r.get("seed") == seed]

    if not records:
        print("No matching records.")
        db.close()
        return

    print(f"{'Timestamp':<28} {'Target':<28} {'Seed':>6} {'LoRA':>5} {'Steps':>5} {'Time':>6} {'Drive Link'}")
    print("-" * 120)
    for r in sorted(records, key=lambda x: x.get("timestamp", "")):
        ts = r.get("timestamp", "")[:19]
        target_name = r.get("target_name", "?")[:27]
        seed_val = r.get("seed", "?")
        lora = r.get("lora_strength", "?")
        steps = r.get("steps", "?")
        gen_time = r.get("generation_time_s", "?")
        gdrive = r.get("gdrive_url", "") or ""
        variant = r.get("variant_index", "")
        if variant != "":
            target_name = f"{target_name} v{variant}"
        print(f"{ts:<28} {target_name:<28} {seed_val:>6} {lora:>5} {steps:>5} {gen_time:>5}s {gdrive}")

    db.close()
    print(f"\n{len(records)} records")


def parse_args():
    p = argparse.ArgumentParser(description="Sync generation metadata from pod to local TinyDB")
    p.add_argument("--host", default="216.81.245.126", help="Pod SSH host")
    p.add_argument("--port", type=int, default=17106, help="Pod SSH port")
    p.add_argument("--key", default="~/.ssh/id_ed25519", help="SSH private key path")
    p.add_argument("--query", action="store_true", help="Query local DB instead of syncing")
    p.add_argument("--target", default=None, help="Filter by target name")
    p.add_argument("--seed", type=int, default=None, help="Filter by seed")
    return p.parse_args()


def main():
    args = parse_args()

    if args.query:
        query_db(target=args.target, seed=args.seed)
    else:
        key_path = str(Path(args.key).expanduser())
        sync_from_pod(args.host, args.port, key_path)


if __name__ == "__main__":
    main()
