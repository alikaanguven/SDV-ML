"""
Run it twice.
The second time if some copies are not there it will update the directory.

REF
---
cp -u, --update:  copy only when the SOURCE file is newer than the destination file or when the destination file is missing
"""

import json
import os
import subprocess


run2_jsons = {"bkg":  ["/eos/vbc/experiments/cms/store/user/aguven/MC_RunIISummer20UL18_20241003.json",
                       "/eos/vbc/experiments/cms/store/user/aguven/MC_RunIISummer20UL17_20241003.json"],
              "data": ["/eos/vbc/experiments/cms/store/user/aguven/Data_production_20240326.json",
                       "/eos/vbc/experiments/cms/store/user/aguven/Data_MET2017.json"],
              "sig":  ["/users/alikaan.gueven/AngPlotter/CMSSW_13_3_0/src/SoftDisplacedVertices/Samples/json/PrivateSignal_v3.json"]}

destination_path = "/scratch-cbe/users/alikaan.gueven/ML_KAAN/run2/"

for tag, jsonfile_list in run2_jsons.items():
    key = "dir" if tag == "sig" else "logical_dir"
    tier = "CustomNanoAODv3" if tag == "sig" else "CustomNanoAOD"

    for jsonfile in jsonfile_list:
        with open(jsonfile,'r') as f:
            json_dict = json.load(f)
        
        for sample, dir in json_dict[tier][key].items(): # e.g. sample = 'zjetstonunuht0200_2018'
            print("Processing", dir)
            physical_dir = "/eos/vbc/experiments/cms" + dir if key == "logical_dir" else dir
            destination_dir = os.path.join(destination_path, sample)
            os.makedirs(destination_dir, exist_ok=True)
            # print(sample)
            # print(f'cp -r -u {physical_dir} {destination_dir}')

            subprocess.Popen(f'cp -r -u {physical_dir} {destination_dir}',
                             shell=True,stdin=None, stdout=None, stderr=None)

