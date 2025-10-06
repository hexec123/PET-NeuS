#!/usr/bin/env python3
import shutil
from pathlib import Path

# Paths
base = Path("public_data/banc")
src = base / "raw_images"
dst = base / "images_4k"

# Make sure directories exist
if not src.exists():
    raise SystemExit(f"❌ Source folder not found: {src}")
dst.mkdir(exist_ok=True)

# Gather raw frames and sort by index
raw_frames = sorted(src.glob("frame_*.png"))
if not raw_frames:
    raise SystemExit(f"❌ No frames found in {src}")

# Clear destination before writing
for f in dst.glob("*.png"):
    f.unlink()

# Copy and rename
for i, f in enumerate(raw_frames):
    out = dst / f"{i:06d}.png"
    shutil.copy2(f, out)

print(f"✅ Restored {len(raw_frames)} images to {dst}")
print(f"Example file: {next(dst.glob('000000.png'))}")
