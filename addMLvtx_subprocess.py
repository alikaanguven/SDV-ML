import ROOT
import pandas as pd
from array import array
import argparse
import os
import shutil
import glob



parser = argparse.ArgumentParser("add a new branch")
parser.add_argument("inputdir",  help="Give us the input directory my precious.", type=str)
parser.add_argument("pred_path", help="Give us the model name my precious.",      type=str)
args = parser.parse_args()


INPUTDIR   = args.inputdir
PRED_PATH = args.pred_path # 'vtx_PART-338_epoch_6'
MODEL_NAME = os.path.basename(PRED_PATH).strip('.parquet')


def checkIfCorrupted(file):
    f = ROOT.TFile(file, 'READ')
    if f.IsZombie():
        print(f'{file} could not processed.')
    else:
        try:
            tree = f.Get("Events")
            _ = tree.GetEntries()
        except Exception as e:
            print('file: ', file)
            print('Exception: \n', e.message, e.args)
    f.Close()
def checkIfBranchExists(tree, branchname):
    for branch in tree.GetListOfBranches():
        if ( branch.GetName() == branchname ):
            print(f'WARNING: Branch {branchname} already exists.')

def main():
    files = glob.glob(f'{INPUTDIR}/**/*.root', recursive=True)
    df = pd.read_parquet(PRED_PATH)
    for file in files:
        print('Starting ', file)
        file_new = file.split('.root')[0] + '_new' + '.root'
        shutil.copyfile(file, file_new)
        print('file: ', file)
        print('file_new: ', file_new)
        
        # Append scores to TTree
        f = ROOT.TFile(file_new, 'UPDATE')
        tree = f.Get("Events")
        pred_scores = array('f', int(tree.GetMaximum('nSDVSecVtx'))*[0.])
        PRED_NAME = MODEL_NAME.replace('-', '_')


        checkIfBranchExists(tree, PRED_NAME)
        branch = tree.Branch(PRED_NAME, pred_scores, f"{PRED_NAME}[nSDVSecVtx]/F")

        n_read = 0
        for i, event in enumerate(tree):
            to_read = event.nSDVSecVtx
            for j in range(to_read):
                pred_scores[j] = df[file][0][n_read+j]
            n_read += to_read
            branch.Fill()
            pred_scores[0] = 0.

        tree.Write("", ROOT.TObject.kOverwrite)
        f.Close()

        checkIfCorrupted(file_new)
        os.remove(file)
        os.rename(file_new, file)
        print('Done writing', file)

if __name__ == "__main__":
    main()