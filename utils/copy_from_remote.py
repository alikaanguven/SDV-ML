#!/usr/bin/env python3

# Usage:
# python3 utils/copy_from_remote.py --json jsons/MLNano.json --remote clip --dest /scratch/agueven/Ang_GNN_nano


import argparse, json, subprocess, os

def run(cmd: str):
    print(f">>> {cmd}")
    proc = subprocess.run(cmd, shell=True)
    if proc.returncode != 0:
        print(f"[warn] command exited with code {proc.returncode}")

def main():
    p = argparse.ArgumentParser(description="Copy .root files from clip to local")
    p.add_argument("--json", required=True, help="Path to the JSON file")
    p.add_argument("--remote", default="clip", help="SSH alias (default: clip)")
    p.add_argument("--dest", required=True, help="Local destination root")
    args = p.parse_args()

    with open(args.json) as f:
        data = json.load(f)

    src_dirs = data["CustomNanoAOD"]["dir"]

    os.makedirs(args.dest, exist_ok=True)

    for key, src in sorted(src_dirs.items()):
        src = src.rstrip("/")
        dst = os.path.join(args.dest.rstrip("/"), key)
        os.makedirs(dst, exist_ok=True)

        # Pull only .root files; resumable; show progress
        cmd = (
            "rsync -a --info=progress2 --partial --append-verify "
            "--include='*/' --include='*.root' --exclude='*' "
            f"{args.remote}:{src}/ {dst}/"
        )
        print(f"\n# {key}\n# src: {src}\n# dst: {dst}")
        run(cmd)

    print("\nDone.")

if __name__ == "__main__":
    main()
