import os
import re
import json
from subprocess import run
import pandas as pd
import utils.root_helpers as root_helpers


INPUTBASE_DIR = "/scratch/agueven/Ang_GNN_nano_merged"
MODEL_NAME = 'vtx_PART-419best_valloss_epoch.parquet'


run2_jsons = {
    "MLNano":  [
        "/home/agueven/SDV-ML/jsons/MLNano.json",
        ]
    }

job_dict = {}
files_per_job = 5

for tag, jsonfile_list in run2_jsons.items():
    tier = "CustomNanoAODv3" if tag == "sig" else "CustomNanoAOD"

    for jsonfile in jsonfile_list:
        with open(jsonfile,'r') as f:
            json_dict = json.load(f)
        
        for sample in json_dict[tier]['dir'].keys():            # e.g. sample = 'zjetstonunuht0200_2018'
            INPUT_DIR = os.path.join(INPUTBASE_DIR, sample)
            PRED_PATH = os.path.join(INPUT_DIR, f'{MODEL_NAME}.parquet')
            df = pd.read_parquet(PRED_PATH)

            files = df.columns
            len_files = len(files)
            chunks = [files[i:i+files_per_job] for i in range(0, len(files), files_per_job)]
            for chunk in chunks:
                INPUT_FILES = ",".join(chunk)
                
                root_helpers.remove_new_root_files(INPUT_DIR)
                command = f'sbatch to_cpu1.sh "python /users/alikaan.gueven/SDV-ML/ParticleTransformer/SDV-ML/addMLvtx_subprocess_v2.py {INPUT_FILES} {PRED_PATH}"' 
                result = run(command, shell=True, capture_output = True, text = True)

                
                job_id = re.search("\d+", result.stdout).group()    # Get the number with '\d+'
                info_dict = {'command': f'sbatch {command}',        # Save command [important for resubmitting]
                            'jobid':   job_id}                      # Save job_id  [identify the status with sacct]
                job_dict[sample] = info_dict                        # Add to dict
                print(result.stdout[:-1])


out_json_path = os.path.join(INPUTBASE_DIR, f'job_ids.json')
print(f"\nWriting to {out_json_path}...\n")
with open(out_json_path, 'w') as f:
    json.dump(job_dict, f)

print('\nFinished. Exiting...')
