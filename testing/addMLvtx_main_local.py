import sys
from pathlib import Path

SCRIPT_DIR  = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
if str(PROJECT_DIR) not in sys.path: sys.path.insert(0, str(PROJECT_DIR))

import os, json, subprocess, concurrent.futures as cf
import pandas as pd
import utils.root_helpers as root_helpers


INPUTBASE_DIR = "/scratch/agueven/Ang_GNN_nano_merged"
MODEL_NAME    = "vtx_PART-419best_valloss_epoch"
RUNNER        = sys.executable
SUBPROC       = str(SCRIPT_DIR / "addMLvtx_subprocess_v2.py")
STATUS_PATH   = os.path.join(INPUTBASE_DIR, "jobs_status.json")

run2_jsons    = {"MLNano": ["/home/agueven/SDV-ML/jsons/MLNano.json"]}
files_per_job = 5
max_workers   = 20


def job_key(sample, chunk_idx):
    return f"{sample}__chunk_{chunk_idx}"


def run_job(cmd):
    cmd = list(map(str, cmd))
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.stdout: print(p.stdout.rstrip())
    if p.stderr: print(p.stderr.rstrip())

    return {
        "status":     "ok" if p.returncode == 0 else "fail",
        "returncode": p.returncode,
        "command":    " ".join(cmd),
        "stdout":     p.stdout[-4000:],
        "stderr":     p.stderr[-4000:],
    }


def save_status(status_dict):
    tmp = STATUS_PATH + ".tmp"
    with open(tmp, "w") as f:
        json.dump(status_dict, f, indent=2)
    os.replace(tmp, STATUS_PATH)



# ---- build job list (skip already ok if resuming) ----

status = {}
if os.path.exists(STATUS_PATH):
    with open(STATUS_PATH) as f:
        status = json.load(f)

jobs = []

for tag, jsonfile_list in run2_jsons.items():
    tier = "CustomNanoAODv3" if tag == "sig" else "CustomNanoAOD"

    for jsonfile in jsonfile_list:
        with open(jsonfile) as f:
            jd = json.load(f)

        for sample in jd[tier]["dir"].keys():
            input_dir = os.path.join(INPUTBASE_DIR, sample)
            pred_path = os.path.join(input_dir, f"{MODEL_NAME}.parquet")
            df = pd.read_parquet(pred_path)

            files  = list(map(str, df.columns))
            chunks = [files[i:i+files_per_job] for i in range(0, len(files), files_per_job)]

            root_helpers.remove_new_root_files(input_dir)

            for ci, chunk in enumerate(chunks):
                key = job_key(sample, ci)
                if status.get(key, {}).get("status") == "ok":
                    continue
                cmd = [RUNNER, SUBPROC, ",".join(chunk), pred_path, "--skip_existing_branch"]
                jobs.append((key, cmd))



# ---- run with a hard cap of 5 in parallel; persist as we go ----

with cf.ThreadPoolExecutor(max_workers=max_workers) as ex:
    futures = {ex.submit(run_job, cmd): key for key, cmd in jobs}
    for fut in cf.as_completed(futures):
        key = futures[fut]
        try:
            res = fut.result()
        except Exception as e:
            res = {"status": "error", "returncode": None, "command": "", "stdout": "", "stderr": str(e)}

        status[key] = res
        print(f"[{res['status']}] {key} (rc={res['returncode']})")
        save_status(status)



# ---- summary ----

ok    = sum(1 for v in status.values() if v.get("status") == "ok")
fail  = sum(1 for v in status.values() if v.get("status") in ("fail", "error"))
print(f"\nDone. ok={ok}, failed={fail}. Status file: {STATUS_PATH}\n")
