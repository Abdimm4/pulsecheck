"""Download the MediaPipe Face Landmarker model and verify its checksum.

Usage:
    python scripts/download_models.py            # saves to models/face_landmarker.task
    python scripts/download_models.py --out path/to/face_landmarker.task

The .task file is kept out of Git (see .gitignore).
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import urllib.request
from pathlib import Path

from pulsecheck.face.landmarker import MODEL_SHA256, MODEL_URL, default_model_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=default_model_path())
    parser.add_argument("--force", action="store_true", help="re-download even if present")
    args = parser.parse_args()

    if args.out.is_file() and not args.force:
        digest = hashlib.sha256(args.out.read_bytes()).hexdigest()
        if digest == MODEL_SHA256:
            print(f"model already present at {args.out}")
            return 0
        print(f"existing file at {args.out} has an unexpected checksum; re-downloading")

    print(f"downloading {MODEL_URL}")
    with urllib.request.urlopen(MODEL_URL, timeout=60) as response:
        data = response.read()
    digest = hashlib.sha256(data).hexdigest()
    if digest != MODEL_SHA256:
        print(f"error: checksum mismatch (got {digest})", file=sys.stderr)
        return 1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.out.with_suffix(".part")
    tmp.write_bytes(data)
    tmp.replace(args.out)
    print(f"saved {len(data)} bytes to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
