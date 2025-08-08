import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix
import awkward as ak
import uproot
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from numba import njit
from numba import jit

import argparse
import random
import datetime
import math
import gc
import glob
import os
import sys

torch.set_printoptions(precision=15)

import ParT_modified as ParT
from vtxLevelDataset2 import ModifiedUprootIterator

import warnings
warnings.filterwarnings("ignore", category=UserWarning)


print('CPU count: ', torch.multiprocessing.cpu_count())

parser = argparse.ArgumentParser("ParT predict")
parser.add_argument("inputdir",  help="Give us the input directory my precious.",  type=str)
parser.add_argument("model_path", help="Give us the model name my precious.",      type=str)
parser.add_argument("-o", "--outputdir", default=None, help="Give us the output directory my precious. Defaults to inputdir.")


args = parser.parse_args()
if args.outputdir is None: args.outputdir = args.inputdir


INPUTDIR   = args.inputdir
MODEL_PATH = args.model_path # 'vtx_PART-338_epoch_6'
MODEL_NAME = os.path.basename(MODEL_PATH).strip('.pt')
OUTPUT_PARQUET_PATH  = os.path.join(args.outputdir,      f'{MODEL_NAME}.parquet')
OUTPUT_PARQUET_PATH2 = os.path.join(args.outputdir,      f'{MODEL_NAME}_logitgap.parquet')



files = glob.glob(f'{INPUTDIR}/**/*.root', recursive=True)
testList = files




branchDict = {}
branchDict['ev'] = ['MET_pt', 'MET_phi', 'nSDVSecVtx', 'Jet_phi', 'Jet_pt', 'Jet_eta', 'nSDVSecVtx']
branchDict['sv'] = ['SDVSecVtx_pt', 'SDVSecVtx_pAngle', 'SDVSecVtx_charge', 'SDVSecVtx_ndof', 'SDVSecVtx_chi2', 'SDVSecVtx_tracksSize', 'SDVSecVtx_sum_tkW', 'SDVSecVtx_LxySig', 'SDVSecVtx_L_phi', 'SDVSecVtx_L_eta', 
                    'SDVIdxLUT_SecVtxIdx', 'SDVIdxLUT_TrackIdx']
branchDict['tk'] = ['SDVTrack_pt', 'SDVTrack_ptError', 'SDVTrack_eta', 'SDVTrack_etaError','SDVTrack_dxy', 'SDVTrack_dxyError', 'SDVTrack_dz', 'SDVTrack_dzError',
                    'SDVTrack_normalizedChi2', 'SDVTrack_eta', 'SDVTrack_phi']
branchDict['label'] = ['SDVSecVtx_matchedLLPnDau_bydau']


shuffle = False
nWorkers = 1
step_size = 3000
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')



model = torch.load(MODEL_PATH, map_location=torch.device(device))

if isinstance(model, torch.nn.DataParallel):
    model = model.module

model.to(device)
model.eval()

def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor """
    return np.eye(num_classes, dtype='uint8')[y]




save_dict  = {}
save_dict2 = {}


print("testList: ", testList)

T = 2.0  # Temperature for softmax scaling

with torch.no_grad():

    for file in testList:
        print("Starting file: ", file)
        logit_bucket  = []
        prob_bucket   = []
        delta_bucket  = []
        print(f'Starting file {file}...')
        testDict = {
            'sig': [file + ':Events'],
            'bkg': []
        }
        testDataset = ModifiedUprootIterator(testDict, branchDict, shuffle=False, nWorkers=1, step_size=step_size)
        testLoader = torch.utils.data.DataLoader(testDataset, num_workers=1, prefetch_factor=1, persistent_workers= True, shuffle=False)
        
        for batch_num, X in enumerate(testLoader):
            if batch_num == 0:
                print('Started batch processes. [test]')
            if batch_num % 100 == 0:
                print('Batch: ', batch_num)
                
            tk_pair_features = torch.cat((X['SDVTrack_px'].permute(1,0,2),
                                          X['SDVTrack_py'].permute(1,0,2),
                                          X['SDVTrack_pz'].permute(1,0,2),
                                          X['SDVTrack_E'].permute(1,0,2),), dim=1)
        
            tk_features = torch.cat((X['SDVTrack_pt'].permute(1,0,2),
                                     X['SDVTrack_ptError'].permute(1,0,2),
                                     X['SDVTrack_eta'].permute(1,0,2),
                                     X['SDVTrack_etaError'].permute(1,0,2),
                                     X['SDVTrack_dxy'].permute(1,0,2),
                                     X['SDVTrack_dxyError'].permute(1,0,2),
                                     X['SDVTrack_dz'].permute(1,0,2),
                                     X['SDVTrack_dzError'].permute(1,0,2),
                                     X['SDVTrack_normalizedChi2'].permute(1,0,2),
                                     torch.cos(X['MET_phi'].permute(1,0)[...,np.newaxis] - X['SDVTrack_phi'].permute(1,0,2))), dim=1)
        
            sv_features = torch.cat((X['SDVSecVtx_pt'],
                                     X['SDVSecVtx_L_eta'],
                                     X['SDVSecVtx_LxySig'],
                                     X['SDVSecVtx_pAngle'],
                                     X['SDVSecVtx_charge'],
                                     X['SDVSecVtx_ndof'],
                                     X['SDVSecVtx_chi2'],
                                     X['SDVSecVtx_tracksSize'],
                                     X['SDVSecVtx_sum_tkW'],
                                     torch.cos(X["MET_phi"] - X["SDVSecVtx_L_phi"]),
                                     torch.cos(X["Jet_phi"] - X["SDVSecVtx_L_phi"]),
                                     torch.cos(X["Jet_eta"] - X["SDVSecVtx_L_eta"])), dim=0).permute(1,0)[..., np.newaxis]
        
                    
            
            tk_pair_features = tk_pair_features.to(device, dtype=float)
            tk_features = tk_features.to(device, dtype=float)
            sv_features = sv_features.to(device, dtype=float)
            
            
            output = model(x=tk_features,
                           v=tk_pair_features,
                           x_sv=sv_features)
            
            logit_bucket.append(output)                     # https://arxiv.org/pdf/1503.02531
            output_softmax = torch.softmax(output, dim=1)   # multiply output by a value if you want temperature sharpening
            prob_bucket.append(output_softmax)

            logit_gap = output[:,1] - output[:,0]
            delta_bucket.append(logit_gap * 1000.)

            # DEBUG stuff...
            # ------------------------------------------------------------
            # if batch_num == 0:
            #     print(X['SDVSecVtx_pt'][0,:5])
            #     print(output)
        
        
        probs_np = torch.cat(prob_bucket)[:,1].cpu().numpy()
        deltas_np = torch.cat(delta_bucket).cpu().numpy()
        save_dict[file]  = np.ascontiguousarray(probs_np)
        save_dict2[file] = np.ascontiguousarray(deltas_np)
        
        gc.collect() # counter memory leaks at the end of each epoch

        
record  = ak.Record(save_dict)
record2 = ak.Record(save_dict2)
ak.to_parquet(record, OUTPUT_PARQUET_PATH)
ak.to_parquet(record2, OUTPUT_PARQUET_PATH2)

