import sys
from pathlib import Path
PROJECT_DIR = Path(__file__).resolve().parents[1]  # parent of "training"
if str(PROJECT_DIR) not in sys.path: sys.path.insert(0, str(PROJECT_DIR))


import ROOT
import utils.root_helpers as root_helpers

import pandas as pd
from array import array
import argparse
import os
import shutil



parser = argparse.ArgumentParser("add a new branch")
parser.add_argument("input_root_files",       help="Input ROOT file to add the new branch to.", type=str)
parser.add_argument("input_pred_path",        help="Input prediction results where the scores will be obtained from.", type=str)
parser.add_argument("--skip_existing_branch", help="Skip if branch exists.", action='store_true')
args = parser.parse_args()


INPUTFILES  = args.input_root_files
PRED_PATH = args.input_pred_path                # '/scratch-cbe/.../vtx_PART-338_epoch_6.parquet'
MODEL_NAME = os.path.basename(PRED_PATH).strip('.parquet')


def main():
    for file in INPUTFILES.split(','):
        PRED_NAME = MODEL_NAME.replace('-', '_')

        df = pd.read_parquet(PRED_PATH)
        print('Starting ', file)
        
        # 1. If branch already exists, remove it

        f = ROOT.TFile(file, 'UPDATE')
        tree = f.Get("Events")
        branch_exists = root_helpers.checkIfBranchExists(tree, PRED_NAME)
        f.Close()
        if branch_exists and not args.skip_existing_branch:
            file_cleaned = file.strip('.root') + '_cleaned.root'
            root_helpers.clone_without_branch(file, file_cleaned, "Events", PRED_NAME)
            root_helpers.checkIfCorrupted(file_cleaned)
            os.remove(file)
            os.rename(file_cleaned, file)
        elif branch_exists and args.skip_existing_branch:
            print(f'Branch {PRED_NAME} already exists in {file}. Skipping...')
            continue
        
        
        # 2. Create a copy of the file and add the new branch

        file_new = file.split('.root')[0] + '_new' + '.root'
        shutil.copyfile(file, file_new)
        print('file: ', file)
        print('file_new: ', file_new)
        
        # Append scores to TTree
        f = ROOT.TFile(file_new, 'UPDATE')
        tree = f.Get("Events")
        pred_scores = array('f', int(tree.GetMaximum('nSDVSecVtx'))*[0.])
        

            
        branch = tree.Branch(PRED_NAME, pred_scores, f"{PRED_NAME}[nSDVSecVtx]/F")

        n_read = 0
        for i, event in enumerate(tree):
            to_read = event.nSDVSecVtx
            for j in range(to_read):
                pred_scores[j] = df[file][0][n_read+j]
            n_read += to_read
            branch.Fill()
            pred_scores[0] = 0.                     #  <----- this might be unnecessary

        tree.Write("", ROOT.TObject.kOverwrite)
        f.Close()

        # 3. If everything is all right delete the old file and rename the new one.

        root_helpers.checkIfCorrupted(file_new)
        os.remove(file)
        os.rename(file_new, file)
        print('Done writing', file)

if __name__ == "__main__":
    main()