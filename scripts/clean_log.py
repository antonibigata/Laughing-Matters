"""
Loop through the log folder and remove all empty folders
"""

import argparse
import shutil
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--log-dir", type=str, default="logs", help="Log directory")
args = parser.parse_args()

log_dir = Path(args.log_dir)

for folder in log_dir.iterdir():
    if folder.is_dir():
        for f in folder.iterdir():
            if "wandb" in f.name:
                break
        else:
            print(f"Removing {folder}")
            shutil.rmtree(folder)
