# scripts/upload_to_drive.py
#
# Uploads generated images to Google Drive using rclone.
# After each upload, fetches the shareable Drive link and writes it
# back into the SQLite feedback database so every rated image has a
# direct Google Drive URL stored alongside its metadata.
#
# Usage:
#   python scripts/upload_to_drive.py
#   python scripts/upload_to_drive.py --dry-run
#   python scripts/upload_to_drive.py --folder "v1_linkedin"

import subprocess
import argparse
import sqlite3
import sys
from datetime import datetime
from pathlib import Path


OUTPUT_DIR = "/runpod-volume/generated"
GDRIVE_REMOTE      = "gdrive"
GDRIVE_BASE_FOLDER = "FluxLoRA/generated"
DB_PATH            = "/runpod-volume/feedback.db"


def parse_args():
    p = argparse.ArgumentParser(description="Upload generated images to Google Drive")
    p.add_argument("--dry-run", action="store_true", help="Preview without uploading")
    p.add_argument("--folder",  default=None, help="Subfolder name (default: date-time)")
    p.add_argument("--src",     default=OUTPUT_DIR, help="Source folder")
    return p.parse_args()


def ensure_gdrive_url_column(conn):
    """Add gdrive_url column to existing databases that predate this feature."""
    columns = [r[1] for r in conn.execute("PRAGMA table_info(generations)").fetchall()]
    if "gdrive_url" not in columns:
        conn.execute("ALTER TABLE generations ADD COLUMN gdrive_url TEXT")
        conn.commit()


def get_drive_link(gdrive_path):
    """
    Uses `rclone link` to get a shareable URL for a file already on Drive.
    Returns the URL string, or None if the command fails.
    """
    result = subprocess.run(
        ["rclone", "link", gdrive_path],
        capture_output=True, text=True
    )
    if result.returncode == 0:
        return result.stdout.strip()
    return None


def upload_and_link(src, subfolder, dry_run=False):
    images = sorted(
        list(Path(src).glob("*.png")) + list(Path(src).glob("*.jpg"))
    )

    if not images:
        print(f"No images found in {src} — nothing to upload.")
        sys.exit(0)

    dst_folder = f"{GDRIVE_REMOTE}:{GDRIVE_BASE_FOLDER}/{subfolder}"

    print(f"Found {len(images)} images")
    print(f"Destination: {dst_folder}")
    if dry_run:
        print("(dry run — no files will be transferred)\n")
        for img in images:
            print(f"  would upload: {img.name}")
        return

    # Open DB — create gdrive_url column if it doesn't exist yet
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    ensure_gdrive_url_column(conn)

    uploaded = 0
    linked   = 0

    for img in images:
        gdrive_file_path = f"{GDRIVE_REMOTE}:{GDRIVE_BASE_FOLDER}/{subfolder}/{img.name}"

        # Upload individual file
        result = subprocess.run(
            ["rclone", "copy", str(img), dst_folder, "--progress"],
            capture_output=False
        )

        if result.returncode != 0:
            print(f"  FAILED to upload: {img.name}")
            continue

        uploaded += 1

        # Fetch the shareable Drive link for this file
        drive_url = get_drive_link(gdrive_file_path)

        if drive_url:
            linked += 1
            # Write back to SQLite — match on filename
            conn.execute(
                "UPDATE generations SET gdrive_url = ? WHERE filename = ?",
                (drive_url, img.name)
            )
            conn.commit()
            print(f"  {img.name}  →  {drive_url}")
        else:
            print(f"  {img.name}  →  uploaded (link unavailable — file may not be rated yet)")

    conn.close()

    print(f"\nUploaded: {uploaded}/{len(images)}")
    print(f"Links written to DB: {linked}")
    if linked < uploaded:
        print("Note: images without DB links were not yet rated.")
        print("      Rate them with feedback.py --session then re-run this script to get links.")


def main():
    args      = parse_args()
    subfolder = args.folder if args.folder else datetime.now().strftime("%Y-%m-%d_%H-%M")
    upload_and_link(args.src, subfolder, dry_run=args.dry_run)


if __name__ == "__main__":
    main()