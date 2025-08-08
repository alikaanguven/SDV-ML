"""

"""
import os
import re
import json
from subprocess import run


def remove_new_root_files(root_dir, dry_run=False):
    """
    Recursively finds and deletes files ending with '_new.root'.
    
    :param root_dir: Path to the top-level directory to clean.
    :param dry_run: If True, just prints the files that *would* be deleted.
    """
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for fname in filenames:
            if fname.endswith('_new.root'):
                full_path = os.path.join(dirpath, fname)
                if dry_run:
                    print('DEBUG remove_new_root_files (dry run): {full_path}')
                try:
                    os.remove(full_path)
                except OSError as e:
                    print(f"Error removing {full_path}: {e}")


INPUTBASE_DIR = "/scratch-cbe/users/alikaan.gueven/ML_KAAN/run2/"
MODEL_NAME = 'vtx_PART-353best_val_epoch'


# run2_jsons = {"bkg":  ["/eos/vbc/experiments/cms/store/user/aguven/MC_RunIISummer20UL18_20241003.json",
#                        "/eos/vbc/experiments/cms/store/user/aguven/MC_RunIISummer20UL17_20241003.json"],
#               "data": ["/eos/vbc/experiments/cms/store/user/aguven/Data_production_20240326.json",
#                        "/eos/vbc/experiments/cms/store/user/aguven/Data_MET2017.json"],
#               "sig":  ["/users/alikaan.gueven/AngPlotter/CMSSW_13_3_0/src/SoftDisplacedVertices/Samples/json/PrivateSignal_v3.json"]}

run2_jsons = {"bkg":  ["/eos/vbc/experiments/cms/store/user/aguven/MC_RunIISummer20UL18_20241003.json",],
              "sig":  ["/users/alikaan.gueven/AngPlotter/CMSSW_13_3_0/src/SoftDisplacedVertices/Samples/json/PrivateSignal_v3.json"]}

job_dict = {}

for tag, jsonfile_list in run2_jsons.items():
    key = "dir" if tag == "sig" else "logical_dir"
    tier = "CustomNanoAODv3" if tag == "sig" else "CustomNanoAOD"

    for jsonfile in jsonfile_list:
        with open(jsonfile,'r') as f:
            json_dict = json.load(f)
        
        for sample in json_dict[tier][key].keys(): # e.g. sample = 'zjetstonunuht0200_2018'
            INPUT_DIR = os.path.join(INPUTBASE_DIR, sample)
            PRED_PATH = os.path.join(INPUT_DIR, f'{MODEL_NAME}.parquet')
            remove_new_root_files(INPUT_DIR, dry_run=True)
            command = f'sbatch to_cpu1.sh "python /users/alikaan.gueven/SDV-ML/ParticleTransformer/SDV-ML/addMLvtx_subprocess.py {INPUT_DIR} {PRED_PATH}"' 
            result = run(command, shell=True, capture_output = True, text = True)

            
            job_id = re.search("\d+", result.stdout).group()    # Get the number with '\d+'
            info_dict = {'command': f'sbatch {command}',        # Save command [important for resubmitting]
                         'jobid':   job_id}                     # Save job_id  [identify the status with sacct]
            job_dict[sample] = info_dict                        # Add to dict
            print(result.stdout[:-1])


out_json_path = os.path.join(INPUTBASE_DIR, f'job_ids.json')
print(f"\nWriting to {out_json_path}...\n")
with open(out_json_path, 'w') as f:
    json.dump(job_dict, f)

print('\nFinished. Exiting...')
