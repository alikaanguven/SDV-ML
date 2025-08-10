#!/usr/bin/env python3
import argparse, json, os, time

ap = argparse.ArgumentParser()
ap.add_argument("--log",    required=True)
ap.add_argument("--jobid",  required=True)
ap.add_argument("--git",    required=True)
ap.add_argument("--cmd",    required=True)
a = ap.parse_args()

os.makedirs(os.path.dirname(a.log), exist_ok=True)
lock = a.log + ".lock"

# acquire a simple lock (atomic create); wait if another submit is writing
while True:
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_RDWR)
        break
    except FileExistsError:
        time.sleep(0.05)

try:
    try:
        with open(a.log, "r") as f:
            data = json.load(f)
    except Exception:
        data = {}

    data[str(a.jobid)] = {
        "git": a.git,
        "job_id": str(a.jobid),
        "cmd": a.cmd
    }

    tmp = a.log + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2, sort_keys=True)
        f.flush(); os.fsync(f.fileno())
    os.replace(tmp, a.log)
finally:
    os.close(fd)
    try: os.unlink(lock)
    except FileNotFoundError: pass
