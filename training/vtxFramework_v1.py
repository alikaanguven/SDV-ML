"""
Usage:       ------
Description: -----
"""
import sys
sys.path.append("..")

import numpy as np
from sklearn.metrics import confusion_matrix



import ParT_modified as ParT
from vtxLevelDataset2 import ModifiedUprootIterator

import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR

import neptune
from neptune.utils import stringify_unsupported

import datetime
import glob
import gc
import math
import warnings

warnings.filterwarnings("ignore", category=UserWarning)


print('CPU count: ', torch.multiprocessing.cpu_count())
# torch.set_num_threads(10)
# torch.set_num_threads(torch.multiprocessing.cpu_count())

def significance(s,b,b_err):
    """
    Median discovery significance
    Definition at slide 33:
    https://www.pp.rhul.ac.uk/~cowan/stat/cowan_munich16.pdf
    
    """
    return np.sqrt(2*((s+b)*np.log(((s+b)*(b+b_err*b_err))/(b*b+(s+b)*b_err*b_err+1e-20)) - 
                    (b*b/(b_err*b_err + 1e-20))*np.log(1+(b_err*b_err*s)/(b*(b+b_err*b_err)+1e-20))))




# MLDATADIR = '/scratch-cbe/users/alikaan.gueven/ML_KAAN/MC2018/all/'
MLDATADIR = '/scratch-cbe/users/alikaan.gueven/ML_KAAN/run2/'
tmpSigList = glob.glob(f'{MLDATADIR}/stop*/**/*.root', recursive=True)
tmpSigList = [sig + ':Events' for sig in tmpSigList]
maxTrain = round(len(tmpSigList)*0.10) # 0.70
maxVal   = round(len(tmpSigList)*0.20) # 1.00

trainSigList = tmpSigList[:maxTrain]
valSigList   = tmpSigList[maxTrain:maxVal]

# trainBkgList = glob.glob(f'{MLDATADIR}/training_set/bkg_mix*.root')
# valBkgList = glob.glob(f'{MLDATADIR}/val_set/bkg_mix*.root')

# trainBkgList = glob.glob('/scratch-cbe/users/alikaan.gueven/ML_KAAN/train/training_set/bkg_mix*.root')
# valBkgList = glob.glob('/scratch-cbe/users/alikaan.gueven/ML_KAAN/train/val_set/bkg_mix*.root')


# trainBkgList = [elm + ':Events' for elm in trainBkgList]
# valBkgList = [elm + ':Events' for elm in valBkgList]


# trainDict = {
#     'sig': trainSigList,
#     'bkg': trainBkgList
# }
# 
# valDict = {
#     'sig': valSigList,
#     'bkg': valBkgList
# }

trainDict = {
    'sig': trainSigList,
    'bkg': None
}

valDict = {
    'sig': valSigList,
    'bkg': None
}



branchDict = {}
branchDict['ev'] = ['MET_phi',
                    'nSDVSecVtx', 
                    'Jet_phi', 'Jet_pt', 'Jet_eta'
                    ]

branchDict['sv'] = ['SDVSecVtx_pt', 
                    'SDVSecVtx_pAngle', 
                    'SDVSecVtx_charge', 
                    'SDVSecVtx_ndof', 
                    'SDVSecVtx_chi2', 
                    'SDVSecVtx_tracksSize', 
                    'SDVSecVtx_sum_tkW', 
                    'SDVSecVtx_LxySig', 
                    'SDVSecVtx_L_phi', 
                    'SDVSecVtx_L_eta', 
                    ]

branchDict['tk'] = ['SDVTrack_pt', 'SDVTrack_ptError', 
                    'SDVTrack_eta', 'SDVTrack_etaError',
                    'SDVTrack_phi', 'SDVTrack_phiError',
                    'SDVTrack_dxy', 'SDVTrack_dxyError', 
                    'SDVTrack_dz', 'SDVTrack_dzError',
                    'SDVTrack_normalizedChi2'
                    ]

branchDict['lut'] = ['SDVIdxLUT_SecVtxIdx', 
                     'SDVIdxLUT_TrackIdx',
                     'SDVIdxLUT_TrackWeight'
                     ]


branchDict['label'] = ['SDVSecVtx_matchedLLPnDau_bydau']


shuffle = True
nWorkers = 2
base_step_size = 150 + 25
if torch.cuda.device_count():
    step_size = base_step_size * torch.cuda.device_count()
else:
    step_size = base_step_size

trainDataset = ModifiedUprootIterator(trainDict, branchDict, shuffle=shuffle, nWorkers=nWorkers, step_size=step_size)
trainLoader = torch.utils.data.DataLoader(trainDataset, num_workers=nWorkers, prefetch_factor=1, persistent_workers= True)


valDataset = ModifiedUprootIterator(valDict, branchDict, shuffle=shuffle, nWorkers=nWorkers, step_size=step_size*5)
valLoader = torch.utils.data.DataLoader(valDataset, num_workers=nWorkers, prefetch_factor=1, persistent_workers= True)



# Training related 
########################################################################

device = torch.device(f'cuda' if torch.cuda.is_available() else 'cpu')

param = {
    "input_dim":     11,
    "input_svdim":   10,
    "num_classes":    2,
    "pair_input_dim": 4,
    "embed_dims": [128, 512, 128],
    "pair_embed_dims": [64, 64, 64],
    "for_inference": False,
    "lr": 8e-4,
    "class_weights": [1, 3],                # [bkg, sig]
    "init_step_size": step_size,
}

# Logging
########################################################################
use_neptune=True

if use_neptune:
    run = neptune.init_run(
        project="alikaan.guven/ParT",
        api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJhNDNjMWJhNS0wMDExLTQ2NzYtOWVjNS1lNzAzOWU4Mzc0MGMifQ==",
    )  # your credentials


if use_neptune:
    run["parameters"] = stringify_unsupported(param)
    run["loader/persistent_workers"] = True


# Training related 
########################################################################
model = ParT.ParticleTransformerDVTagger(input_dim=param['input_dim'],
                                         input_svdim=param['input_svdim'],
                                         num_classes=param['num_classes'],
                                         pair_input_dim=param['pair_input_dim'],
                                         embed_dims=param['embed_dims'],
                                         for_inference=param['for_inference'])

if torch.cuda.device_count() > 1:
    print("Using ", torch.cuda.device_count(), "GPUs!")
    model = nn.DataParallel(model)
model.to(device, dtype=float)


optimizer = torch.optim.Adam(model.parameters(), lr=param['lr'])

scheduler = StepLR(optimizer, step_size=3, gamma=0.75)
criterion = nn.CrossEntropyLoss(reduction='none')


def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor """
    return np.eye(num_classes, dtype='uint8')[y]


epochs = 40

print('Starting train...')
best_val_acc = 0
for epoch in range(epochs):
    model.train()
    print('Epoch ', epoch)
    # Logging
    ########################################################################
    losses = []
    TP_epoch = 0
    FN_epoch = 0
    FP_epoch = 0
    TN_epoch = 0
    print(trainLoader.dataset)
    if use_neptune:
        run['parameters/step_size'].append(trainLoader.dataset.step_size)
    for batch_num, X in enumerate(trainLoader):
        if batch_num == 0:
            print('Started batch processes. [train]')
            # print(batch_num, X['SDVSecVtx_pAngle'].shape)
        
        # Training related 
        ########################################################################
        
        optimizer.zero_grad()
    
    
        # Preprocess input
        ########################################################################
        tk_pair_features = torch.cat((X['SDVTrack_px'],
                                      X['SDVTrack_py'],
                                      X['SDVTrack_pz'],
                                      X['SDVTrack_E']), dim=0).permute(1,0,2)


        tk_features = torch.cat((X['SDVTrack_pt'],  X['SDVTrack_ptError'],
                                 X['SDVTrack_eta'], X['SDVTrack_etaError'],
                                 X['SDVTrack_dxy'], X['SDVTrack_dxyError'],
                                 X['SDVTrack_dz'],  X['SDVTrack_dzError'],
                                 X['SDVTrack_normalizedChi2'],
                                 X['SDVIdxLUT_TrackWeight'],
                                 torch.cos(X['MET_phi'][...,np.newaxis] - X['SDVTrack_phi'])), dim=0).permute(1,0,2)
        
        sv_features = torch.cat((X['SDVSecVtx_pt'],
                                 X['SDVSecVtx_L_eta'],
                                 X['SDVSecVtx_LxySig'],
                                 X['SDVSecVtx_pAngle'],
                                 X['SDVSecVtx_charge'],
                                 X['SDVSecVtx_chi2'] / X['SDVSecVtx_ndof'],
                                 X['SDVSecVtx_sum_tkW'] / X['SDVSecVtx_tracksSize'],
                                 torch.cos(X["MET_phi"] - X["SDVSecVtx_L_phi"]),
                                 torch.cos(X["Jet_phi"] - X["SDVSecVtx_L_phi"]),
                                 torch.abs(X["Jet_eta"] - X["SDVSecVtx_L_eta"]),
                                 ), dim=0).permute(1,0)[..., np.newaxis]
        
        torch.nan_to_num(tk_pair_features, nan=-999., out=tk_pair_features)
        torch.nan_to_num(tk_features,      nan=-999., out=tk_features)
        torch.nan_to_num(sv_features,      nan=-999., out=sv_features)

        # print(tk_features.shape)
        # print(tk_features)


        label = X['SDVSecVtx_matchedLLPnDau_bydau'].permute(1,0)
        y = label[:,0]
                
        
        tk_pair_features = tk_pair_features.to(device, dtype=float)
        tk_features = tk_features.to(device, dtype=float)
        sv_features = sv_features.to(device, dtype=float)

        isSignal = (y > 1) # matchedLLPnDau_bydau
        isSignal = isSignal.to(dtype=int)
        isSignal = to_categorical(isSignal.numpy(), 2)
        y = torch.tensor(isSignal).to(device, dtype=float)  
   
        # Training related 
        ########################################################################
        output = model(x=tk_features,
                       v=tk_pair_features,
                       x_sv=sv_features)
        
        # Setting the weights with predetermined class inbalance
        sample_weights = torch.sum((y==1) * torch.tensor(param['class_weights']).to(device, dtype=float),axis=-1)

        # Class inbalance is determined during training.
        # sigWeight = torch.sum(y.data[:,-1] == 0) / torch.sum(y.data[:,-1] == 1)
        # sample_weights = torch.sum((y==1) * torch.tensor([1, sigWeight]).to(device, dtype=float),axis=-1)

        loss = criterion(output, y)
        loss = torch.mean(sample_weights * loss)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        
        output = torch.softmax(output, dim=1)
        
        sigThreshold = 0.98
        y_pred01 = (output[:,-1] > sigThreshold).to('cpu', dtype=int)
        y_test01 = y.data[:,-1].to('cpu', dtype=int)
        CM = confusion_matrix(y_test01, y_pred01)
        TP = CM[1,1]
        FN = CM[1,0]
        FP = CM[0,1]
        TN = CM[0,0]

        TP_epoch += TP
        FN_epoch += FN
        FP_epoch += FP
        TN_epoch += TN

            
        TPR = TP / (TP+FN) if (TP+FN) != 0 else 0
        PPV = TP / (TP+FP) if (TP+FP) != 0 else 0

        if use_neptune:
            run["train/TPR"].append(TPR) # if math.isfinite(TPR) else 0.
            run["train/PPV"].append(PPV) # if math.isfinite(PPV) else 0.
        
        elif batch_num %10 == 0:
            print('batch_num: ', batch_num)
            print('Class imbalance: ', [round((torch.sum(y.data[:,-1] == 0) / torch.sum(y.data[:,-1] == 1)).item(),2), 1]) # bkg/sig
            print('#'*80)
            print('#'*80)
            print('TPR: ', TPR)
            print('PPV: ', PPV)
            print('CM:')
            print(CM)
            
        
        
        # print('y: ', y)
        # print('sample_weights: ', sample_weights)
        # print('output: ', output)
        # print('torch.sum(y[:,0]==1): ', torch.sum(y[:,0]==1))
        # print('torch.sum(y[:,1]==1): ', torch.sum(y[:,1]==1))
        
        # break

    
        # Logging
        ########################################################################
        
        acc = (TP+TN) / (TP + FN + FP + TN)
        if use_neptune:
            run["train/accuracy_batch"].append(acc) if math.isfinite(acc) else print('train/accuracy_batch is not finite')
            run["train/loss_batch"].append(loss.item()) if math.isfinite(loss.item()) else print('train/loss_batch is not finite')
            
        else:
            if batch_num %10 == 0:
                print('Acc:  ', acc.item())
                print('Loss: ', loss.item())
    

    acc_epoch = (TP_epoch+TN_epoch) / (TP_epoch + FN_epoch + FP_epoch + TN_epoch)
    TPR_epoch = TP_epoch / (TP_epoch+FN_epoch)
    PPV_epoch = TP_epoch / (TP_epoch+FP_epoch)
    loss_epoch = sum(losses)/len(losses)
    if use_neptune:
        run["train/accuracy_epoch"].append(acc_epoch) if math.isfinite(acc_epoch) else print('train/accuracy_epoch is not finite')
        run["train/losses_epoch"].append(loss_epoch) if math.isfinite(loss_epoch) else print('train/losses_epoch is not finite')
        run["train/TPR_epoch"].append(TPR_epoch) if math.isfinite(TPR_epoch) else print('train/TPR_epoch is not finite')
        run["train/PPV_epoch"].append(PPV_epoch) if math.isfinite(PPV_epoch) else print('train/PPV_epoch is not finite')
    else:
        print('Acc  [epoch]: ', acc_epoch)
        print('Loss [epoch]: ', loss_epoch)

    
    # Validation related 
    ########################################################################
    print('\n'*2)
    print('Entering validation phase...')
    losses = []
    TP_epoch = 0
    FN_epoch = 0
    FP_epoch = 0
    TN_epoch = 0
    output_bucket = []
    label_bucket = []
    print("Before eval")
    model.eval()
    print("After eval")
    with torch.no_grad():
        print("torch.no_grad()")
        for batch_num, X in enumerate(valLoader):
            if batch_num == 0:
                print('Started batch processes. [validation]')
                
            tk_pair_features = torch.cat((X['SDVTrack_px'],
                                      X['SDVTrack_py'],
                                      X['SDVTrack_pz'],
                                      X['SDVTrack_E']), dim=0).permute(1,0,2)


            tk_features = torch.cat((X['SDVTrack_pt'],  X['SDVTrack_ptError'],
                                    X['SDVTrack_eta'], X['SDVTrack_etaError'],
                                    X['SDVTrack_dxy'], X['SDVTrack_dxyError'],
                                    X['SDVTrack_dz'],  X['SDVTrack_dzError'],
                                    X['SDVTrack_normalizedChi2'],
                                    X['SDVIdxLUT_TrackWeight'],
                                    torch.cos(X['MET_phi'][...,np.newaxis] - X['SDVTrack_phi'])), dim=0).permute(1,0,2)
            
            sv_features = torch.cat((X['SDVSecVtx_pt'],
                                    X['SDVSecVtx_L_eta'],
                                    X['SDVSecVtx_LxySig'],
                                    X['SDVSecVtx_pAngle'],
                                    X['SDVSecVtx_charge'],
                                    X['SDVSecVtx_chi2'] / X['SDVSecVtx_ndof'],
                                    X['SDVSecVtx_sum_tkW'] / X['SDVSecVtx_tracksSize'],
                                    torch.cos(X["MET_phi"] - X["SDVSecVtx_L_phi"]),
                                    torch.cos(X["Jet_phi"] - X["SDVSecVtx_L_phi"]),
                                    torch.abs(X["Jet_eta"] - X["SDVSecVtx_L_eta"]),
                                    ), dim=0).permute(1,0)[..., np.newaxis]
            
            torch.nan_to_num(tk_pair_features, nan=-999., out=tk_pair_features)
            torch.nan_to_num(tk_features,      nan=-999., out=tk_features)
            torch.nan_to_num(sv_features,      nan=-999., out=sv_features)


            label = X['SDVSecVtx_matchedLLPnDau_bydau'].permute(1,0)
            y = label


            tk_pair_features = tk_pair_features.to(device, dtype=float)
            tk_features = tk_features.to(device, dtype=float)
            sv_features = sv_features.to(device, dtype=float)

            y = y.to(device, dtype=float)

            ymaxSig = torch.max(y > 1, axis=-1).values
            ymaxSig = ymaxSig.float()
            yBkg = (ymaxSig != 1).float()
            y = torch.concatenate((yBkg[:, np.newaxis], ymaxSig[:, np.newaxis]), axis=-1)
            output = model(x=tk_features,
                           v=tk_pair_features,
                           x_sv=sv_features)

            # Setting the weights with predetermined class inbalance
            sample_weights = torch.sum((y==1) * torch.tensor(param['class_weights']).to(device, dtype=float),axis=-1)

            # Class inbalance is determined during training.
            # sigWeight = torch.sum(y.data[:,-1] == 0) / torch.sum(y.data[:,-1] == 1)
            # sample_weights = torch.sum((y==1) * torch.tensor([1, sigWeight]).to(device, dtype=float),axis=-1)

            loss = criterion(output, y)
            loss = torch.mean(sample_weights * loss)
            losses.append(loss.item())
            output = torch.softmax(output, dim=1)


            output_bucket.append(output)
            label_bucket.append(y.data)


            sigThreshold = 0.98
            y_pred01 = (output[:,-1] > sigThreshold).to('cpu', dtype=int)
            y_test01 = y.data[:,-1].to('cpu', dtype=int)
            CM = confusion_matrix(y_test01, y_pred01)
            TP = CM[1,1]
            FN = CM[1,0]
            FP = CM[0,1]
            TN = CM[0,0]

            TP_epoch += TP
            FN_epoch += FN
            FP_epoch += FP
            TN_epoch += TN

            if batch_num %10 == 0:

                TPR = TP / (TP+FN) if (TP+FN) != 0 else 0
                PPV = TP / (TP+FP) if (TP+FP) != 0 else 0

                if use_neptune:
                    run["val/TPR"].append(TPR) # if math.isfinite(TPR) else 0.
                    run["val/PPV"].append(PPV) # if math.isfinite(PPV) else 0.
                else:
                    print('batch_num: ', batch_num)
                    print('Class imbalance: ', [1, round((torch.sum(y.data[:,-1] == 0) / torch.sum(y.data[:,-1] == 1)).item(),2)])
                    print('#'*80)
                    print('#'*80)
                    print('TPR: ', TPR)
                    print('PPV: ', PPV)
                    print('CM:')
                    print(CM)




            acc = (TP+TN) / (TP + FN + FP + TN)
            if use_neptune:
                run["val/accuracy_batch"].append(acc) if math.isfinite(acc) else print('val/accuracy_batch is not finite')
                run["val/loss_batch"].append(loss.item()) if math.isfinite(loss.item()) else print('val/loss_batch is not finite')
            else:
                if batch_num %10 == 0:
                    print('Acc:  ', acc.item())
                    print('Loss: ', loss.item())
        
            
        acc_epoch = (TP_epoch+TN_epoch) / (TP_epoch + FN_epoch + FP_epoch + TN_epoch)
        TPR_epoch = TP_epoch / (TP_epoch+FN_epoch)
        PPV_epoch = TP_epoch / (TP_epoch+FP_epoch)
        loss_epoch = sum(losses)/len(losses)
        
        bkg_to_sigcalc  = 1e4 / (FP_epoch + TN_epoch) * FP_epoch  # 10000 * FPR
        sig_to_sigcalc1 = 200 / (TP_epoch + FN_epoch) * TP_epoch  #   200 * TPR
        sig_to_sigcalc2 = 15  / (TP_epoch + FN_epoch) * TP_epoch  #    15 * TPR
        
        signif1_epoch = significance(sig_to_sigcalc1, bkg_to_sigcalc, bkg_to_sigcalc * 0.20)
        signif2_epoch = significance(sig_to_sigcalc2, bkg_to_sigcalc, bkg_to_sigcalc * 0.20)
        if use_neptune:
            run["val/accuracy_epoch"].append(acc_epoch) if math.isfinite(acc_epoch) else print('val/accuracy_epoch is not finite')
            run["val/losses_epoch"].append(loss_epoch) if math.isfinite(loss_epoch) else print('val/losses_epoch is not finite')
            run["val/TPR_epoch"].append(TPR_epoch) if math.isfinite(TPR_epoch) else print('val/TPR_epoch is not finite')
            run["val/PPV_epoch"].append(PPV_epoch) if math.isfinite(PPV_epoch) else print('val/PPV_epoch is not finite')
            run["val/signif600_epoch"].append(signif1_epoch) if math.isfinite(signif1_epoch) else print('val/signif600_epoch is not finite')
            run["val/signif1000_epoch"].append(signif2_epoch) if math.isfinite(signif2_epoch) else print('val/signif1000_epoch is not finite')
        else:
            print('Acc  [epoch]: ', acc_epoch)
            print('Loss [epoch]: ', loss_epoch)

        if acc_epoch > best_val_acc:
            best_val_acc = acc_epoch
            savename = None
            if use_neptune:
                savename = run["sys/id"].fetch() + 'best_val_epoch.pt'
            else:
                savename = 'ParT_modified' + datetime.datetime.now().strftime('_%Y-%m-%d-%H-%M-%S_') + 'best_val_epoch.pt'
            # torch.save(model.state_dict(), '/users/alikaan.gueven/ParticleTransformer/PyTorchExercises/models/vtx_' + savename)
            torch.save(model, '/groups/hephy/cms/alikaan.gueven/ParT/models/vtx_' + savename)
        
        if use_neptune:
            torch.save(model, '/groups/hephy/cms/alikaan.gueven/ParT/models/vtx_' + run["sys/id"].fetch() + '_epoch_' + str(epoch) + '.pt')
        else:
            torch.save(model, '/groups/hephy/cms/alikaan.gueven/ParT/models/vtx_' + datetime.datetime.now().strftime('_%Y-%m-%d-%H-%M-%S_') + '_epoch_' + str(epoch) + '.pt')


        if use_neptune:
            isMatched = torch.cat(label_bucket)[:,1] == 1
            binVals1, binEdges1 = np.histogram(torch.cat(output_bucket)[:,1][isMatched].cpu(), np.arange(-0.01,1.01,0.02))
            binCenters1 = (binEdges1[:-1] + binEdges1[1:]) / 2

            binVals0, binEdges0 = np.histogram(torch.cat(output_bucket)[:,1][~isMatched].cpu(), np.arange(-0.01,1.01,0.02))
            binCenters0 = (binEdges0[:-1] + binEdges0[1:]) / 2
            
            plt.figure(figsize=(10,8))
            plt.errorbar(binCenters1, binVals1, np.sqrt(binVals1), fmt='o', capthick=1, capsize=3, color='indigo', markersize=5, label="matched")
            plt.errorbar(binCenters0, binVals0, np.sqrt(binVals0), fmt='o', capthick=1, capsize=3, color='red', markersize=5, label="unmatched")
            plt.ylim(1, 2*(sum(binVals1)+sum(binVals0)))
            plt.yscale('log')
            plt.title('Validation Dataset')
            plt.xlabel('ParT Score')
            plt.ylabel('vtx Count')
            plt.legend()
            plt.savefig('hist.png')

            run['score_hist'].append(neptune.types.File('hist.png'))

    
    gc.collect() # counter memory leaks at the end of each epoch
    
if use_neptune:
    run.stop()




