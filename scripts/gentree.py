"""Generate a collapsible directory tree for the repo.

Rules:
  - Video dirs (>5 files)  -> "[438 videos: mp4, webm]" summary
  - Image dirs (>10 files) -> "[200 frames: jpg, png]" summary
  - Everything else listed individually (model weights, JSONs, CSVs, configs)
  - Skip .git, __pycache__, .pyc, node_modules
"""
import os
import sys
from collections import Counter

VIDEO_EXT = {'.mp4', '.avi', '.mov', '.webm', '.mkv', '.flv', '.wmv', '.m4v', '.mpg', '.mpeg'}
IMAGE_EXT = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp', '.tiff', '.tif'}
SKIP_DIRS = {'.git', '__pycache__', 'node_modules', '.mypy_cache', '.pytest_cache'}
SKIP_EXTS = {'.pyc', '.pyo', '.metadata'}

VIDEO_COLLAPSE = 5
IMAGE_COLLAPSE = 10

lines_out = []

def walk(path, prefix=""):
    try:
        entries = sorted(os.listdir(path))
    except PermissionError:
        return

    dirs = [e for e in entries if os.path.isdir(os.path.join(path, e))]
    files = [e for e in entries if os.path.isfile(os.path.join(path, e))]

    videos, images, others = [], [], []
    for f in files:
        ext = os.path.splitext(f)[1].lower()
        if ext in SKIP_EXTS:
            continue
        if ext in VIDEO_EXT:
            videos.append(f)
        elif ext in IMAGE_EXT:
            images.append(f)
        else:
            others.append(f)

    for i, d in enumerate(dirs):
        if d in SKIP_DIRS:
            continue
        remaining = (len(dirs) - i - 1) + len(videos) + len(images) + len(others)
        is_last = remaining == 0
        c = "`-- " if is_last else "|-- "
        lines_out.append(f"{prefix}{c}{d}/")
        new_prefix = prefix + ("    " if is_last else "|   ")
        walk(os.path.join(path, d), new_prefix)

    all_items = []
    if len(videos) > VIDEO_COLLAPSE:
        exts = Counter(os.path.splitext(v)[1].lower() for v in videos)
        summary = ", ".join(f"{c} {e}" for e, c in sorted(exts.items()))
        all_items.append(f"[{len(videos)} videos: {summary}]")
    else:
        all_items.extend(videos)

    if len(images) > IMAGE_COLLAPSE:
        exts = Counter(os.path.splitext(v)[1].lower() for v in images)
        summary = ", ".join(f"{c} {e}" for e, c in sorted(exts.items()))
        all_items.append(f"[{len(images)} frames: {summary}]")
    else:
        all_items.extend(images)

    all_items.extend(others)
    all_items.sort()

    for i, item in enumerate(all_items):
        is_last = (i == len(all_items) - 1)
        c = "`-- " if is_last else "|-- "
        lines_out.append(f"{prefix}{c}{item}")


if __name__ == "__main__":
    root = sys.argv[1] if len(sys.argv) > 1 else "."
    os.chdir(root)
    walk(".")
    for line in lines_out:
        print(line)
