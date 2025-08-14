#!/usr/bin/env python3
# Usage:
# python3 utils/push_to_remote.py --src /scratch/agueven/Ang_GNN_nano --remote clip --dest /scratch-cbe/users/alikaan.gueven/ML_KAAN/Ang_GNN_files

import argparse, subprocess

def run(cmd: str):
    print(f">>> {cmd}")
    proc = subprocess.run(cmd, shell=True)
    if proc.returncode != 0:
        print(f"[warn] command exited with code {proc.returncode}")

def main():
    ap = argparse.ArgumentParser(description="Copy all files/dirs from local to a remote host via rsync.")
    ap.add_argument("--src", default="/scratch/agueven/Ang_GNN_nano", help="Local source directory")
    ap.add_argument("--remote", default="clip", help="SSH alias (default: clip)")
    ap.add_argument("--dest", default="/scratch-cbe/users/alikaan.gueven/ML_KAAN/Ang_GNN_files",
                    help="Remote destination directory")
    args = ap.parse_args()

    # Ensure remote destination exists
    run(f'ssh {args.remote} "bash -lc \'mkdir -p {args.dest}\'"')

    # Copy the CONTENTS of src into dest (note trailing slash on src!)
    src = args.src.rstrip("/") + "/"
    cmd = (
        "rsync -a --info=progress2 --partial --append-verify "
        f"{src} {args.remote}:{args.dest}/"
    )
    run(cmd)
    print("\nDone.")

if __name__ == "__main__":
    main()
