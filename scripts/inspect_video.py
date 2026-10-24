"""Print a video's measured metadata and optionally save sample frames.

Usage:
    python scripts/inspect_video.py path/to/video.mp4
    python scripts/inspect_video.py path/to/video.mp4 --extract out_dir --every 30
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2

from pulsecheck.video import VideoDecodeError, iter_frames, probe_video


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("video", type=Path)
    parser.add_argument("--extract", type=Path, help="directory to save sample frames as PNG")
    parser.add_argument("--every", type=int, default=30, help="save every Nth frame (default 30)")
    args = parser.parse_args()

    try:
        meta = probe_video(args.video)
    except VideoDecodeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(json.dumps(meta.to_dict(), indent=2))

    if args.extract:
        args.extract.mkdir(parents=True, exist_ok=True)
        saved = 0
        for frame in iter_frames(args.video):
            if frame.index % args.every == 0:
                name = f"frame_{frame.index:05d}_{frame.timestamp_s:.3f}s.png"
                cv2.imwrite(str(args.extract / name), frame.image)
                saved += 1
        print(f"saved {saved} frames to {args.extract}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
