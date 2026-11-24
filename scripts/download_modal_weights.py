#!/usr/bin/env python3
"""
Downloads trained LoRA adapters or merged 16-bit models from the Modal volume
to your local machine.

Usage:
  python scripts/download_modal_weights.py --model qwen --target-dir models/qwen-dutch-tutor
"""

import argparse
import subprocess
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="qwen", choices=["qwen", "qwen3", "gemma", "gemma3"])
    p.add_argument("--type", default="merged", choices=["merged", "adapters"], help="Download full merged model or adapters only")
    p.add_argument("--target-dir", type=Path, default=Path("models/exported"))
    args = p.parse_args()

    tag_map = {
        "qwen": "Qwen_Qwen2.5-0.5B-Instruct",
        "qwen3": "Qwen_Qwen3-0.6B",
        "gemma": "google_gemma-3-1b-it",
        "gemma3": "google_gemma-3-1b-it",
    }
    clean_tag = tag_map[args.model]
    remote_path = f"{args.type}/{clean_tag}"
    args.target_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {remote_path} from Modal volume 'dutch-tutor-models' -> {args.target_dir}...")
    cmd = [
        "modal", "volume", "get",
        "dutch-tutor-models",
        remote_path,
        str(args.target_dir),
    ]
    try:
        subprocess.run(cmd, check=True)
        print(f"✅ Download complete: {args.target_dir}")
    except FileNotFoundError:
        print("FAIL: 'modal' CLI is not installed or not in PATH.")
        print("Install modal via: pip install modal && modal setup")
    except subprocess.CalledProcessError as e:
        print(f"FAIL: modal volume get failed ({e})")


if __name__ == "__main__":
    main()
