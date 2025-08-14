from subprocess import run, PIPE
import re
import json
import os

import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]  # parent of "training"
if str(PROJECT_DIR) not in sys.path: sys.path.insert(0, str(PROJECT_DIR))


# Get sample names from JSON file
# -----------------------------------
JSON_PATH = '/home/agueven/SDV-ML/jsons/MLNano.json'
with open(JSON_PATH) as f:
    x = json.load(f)
samples = x['CustomNanoAOD']['dir'].keys()


PREDICT_SCRIPT = '/home/agueven/SDV-ML/testing/vtxFramework_v2_predict.py'
MODEL_PATH     = '/scratch/agueven/ParT_saved_models/vtx_PART-419best_valloss_epoch.pt'
INPUT_BASEDIR  = '/scratch/agueven/Ang_GNN_nano_merged/'
job_dict = {}


for sample in samples:
    print("Starting sample: ", sample)
    INPUT_DIR = os.path.join(INPUT_BASEDIR, sample) # e.g. sample='zjetstonunuht0800_2017'

    command = f'python3 {PREDICT_SCRIPT} {INPUT_DIR} {MODEL_PATH}'
    result = run(command, shell=True, capture_output = True, text = True)
    # job_id = re.search("\d+", result.stdout).group()    # Get the number with '\d+'
    # info_dict = {'command': f'sbatch {command}',        # Save command [important for resubmitting]
    #              'jobid':   job_id}                     # Save job_id  [identify the status with sacct]
    # job_dict[sample] = info_dict                        # Add to dict
    print('stdout...')
    print('-'*80)
    print(result.stdout[:-1])
    print('stderr...')
    print('-'*80)
    print(result.stderr[:-1])

study = 'MC_RunIISummer20UL18'
out_json_path = os.path.join(INPUT_BASEDIR, f'job_ids_{study}.json')
print(f"\nWriting to {out_json_path}...\n")
with open(out_json_path, 'w') as f:
    json.dump(job_dict, f)

print('\nFinished. Exiting...')