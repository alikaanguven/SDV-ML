#!/usr/bin/env python3
import os, glob, subprocess, shutil
import ROOT

SRC_BASE   = "/scratch/agueven/Ang_GNN_nano"
DST_BASE   = "/scratch/agueven/Ang_GNN_nano_merged"

CHUNK_GB   = 4.0
CHUNK_BYTES= int(CHUNK_GB * 1024**3)
HADD_J     = "1"   # threads for hadd


for sample in sorted(os.listdir(SRC_BASE)):
    if sample != "qcdht0700_2018": continue
    src_dir = os.path.join(SRC_BASE, sample)
    if not os.path.isdir(src_dir):
        continue
    files = sorted(f for f in glob.glob(os.path.join(src_dir, "*.root"))
                   if "merged" not in os.path.basename(f))
    if not files:
        continue
    # make output dirs
    local_merge_dir = os.path.join(src_dir, "merged_4GB")
    os.makedirs(local_merge_dir, exist_ok=True)

    dst_dir = os.path.join(DST_BASE, sample)
    os.makedirs(dst_dir, exist_ok=True)

    # greedy chunking to ~4 GiB
    chunks, cur, cur_size = [], [], 0
    for f in files:
        s = os.path.getsize(f)
        if cur and cur_size + s > CHUNK_BYTES:
            chunks.append(cur)
            cur, cur_size = [], 0
        cur.append(f)
        cur_size += s
    if cur:
        chunks.append(cur)

    print(f"{sample}: {len(files)} files → {len(chunks)} merged outputs (~{CHUNK_GB} GiB)")

    # merge and copy
    for i, chunk in enumerate(chunks):
        out_name = f"{sample}__merged.root" if len(chunks) == 1 else f"{sample}__merged_{i:03d}.root"
        local_out = os.path.join(local_merge_dir, out_name)
        dst_out   = os.path.join(dst_dir, out_name)

        cmd = ["hadd", "-fkT", local_out] + chunk
        res = subprocess.run(cmd)

        if res.returncode != 0:
            print(f"  !! hadd failed ({out_name}) with code {res.returncode}")
            continue
        
        # delete tags ...
        f = ROOT.TFile.Open(local_out, "UPDATE")
        if f and not f.IsZombie():
            f.Delete("tag;*")
            f.Write()   # persist metadata changes
            f.Close()
            print(f"[cleaned tags] {local_out}")
        else:
            print(f"[warn] could not open {local_out} for cleaning")


        shutil.copy2(local_out, dst_out)
        os.remove(local_out)
        print(f"  -> {out_name}  (copied to {dst_dir})")