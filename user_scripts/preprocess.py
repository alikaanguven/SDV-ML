import sys
sys.path.append("/users/alikaan.gueven/SDV-ML/ParticleTransformer/SDV-ML")

import torch
import awkward as ak
import numpy as np
import utils.help_preprocess as hp


def transform(batch, branch_dict):
    """
    Preprocess the input data for training or evaluation.
    
    Args:
        
        
    Returns:
        
    """
    out_dict = {}
    X = ak.concatenate(batch, axis=0)
    # X = batch[0]

    X['jet0_phi'] = X['Jet_phi'][:, 0]
    X['jet0_eta'] = X['Jet_eta'][:, 0]

    Jet_isTight = ak.values_astype(X['Jet_jetId'] & (1 << 1), bool)
    jet_sel_vetomapsel = (Jet_isTight &
                         (X['Jet_pt'] > 15) &
                         ((X['Jet_chEmEF'] + X['Jet_neEmEF']) < 0.9) &
                         (X['Jet_muonIdx1'] == -1) &
                         (X['Jet_muonIdx2'] == -1)
                         )
    Jet_phi_vetomapsel = X.Jet_phi[jet_sel_vetomapsel]
    Jet_eta_vetomapsel = X.Jet_eta[jet_sel_vetomapsel]
    X['Jet_phi_vetomapsel'] = Jet_phi_vetomapsel
    X['Jet_eta_vetomapsel'] = Jet_eta_vetomapsel


    min_dR, min_dphi, min_deta = hp.calc_dR_deta_dphi(X['SDVSecVtx_L_phi'], X['SDVSecVtx_L_eta'], X['Jet_phi_vetomapsel'], X['Jet_eta_vetomapsel'])
    X['SDVSecVtx_closestJetdR']   = min_dR
    # X['SDVSecVtx_closestJetdphi'] = min_dphi
    # X['SDVSecVtx_closestJetdeta'] = min_deta

    branch_dict['ev'].append('jet0_phi')
    branch_dict['ev'].append('jet0_eta')
    branch_dict['sv'].append('SDVSecVtx_closestJetdR')
    # branch_dict['sv'].append('SDVSecVtx_closestJetdphi')
    # branch_dict['sv'].append('SDVSecVtx_closestJetdeta')
    
    X = hp.pad_and_fill(X, branch_dict, svDim=12, tkDim=10, fillValue=float("nan"))


    for key in X.keys(): X[key] = X[key].unsqueeze(0)
    out_dict["tk_pair_features"] = torch.cat((X['SDVTrack_px'],
                                              X['SDVTrack_py'],
                                              X['SDVTrack_pz'],
                                              X['SDVTrack_E']), dim=0).permute(1,0,2)


    out_dict["tk_features"] = torch.cat((X['SDVTrack_pt'],  X['SDVTrack_ptError'],
                                         X['SDVTrack_eta'], X['SDVTrack_etaError'],
                                         X['SDVTrack_dxy'], X['SDVTrack_dxyError'],
                                         X['SDVTrack_dz'],  X['SDVTrack_dzError'],
                                         X['SDVTrack_normalizedChi2'],
                                         X['SDVIdxLUT_TrackWeight'],
                                         torch.cos(X['MET_phi'][...,np.newaxis] - X['SDVTrack_phi'])), dim=0).permute(1,0,2)
    
    
    
    out_dict["sv_features"] = torch.cat((X['SDVSecVtx_pt'],
                                         X['SDVSecVtx_L_eta'],
                                         X['SDVSecVtx_LxySig'],
                                         X['SDVSecVtx_pAngle'],
                                         X['SDVSecVtx_charge'],
                                         X['SDVSecVtx_chi2'] / X['SDVSecVtx_ndof'],
                                         X['SDVSecVtx_sum_tkW'] / X['SDVSecVtx_tracksSize'],
                                         X['SDVSecVtx_closestJetdR'],
                                         torch.cos(X["MET_phi"] - X["SDVSecVtx_L_phi"]),
                                         torch.cos(X["jet0_phi"] - X["SDVSecVtx_L_phi"]),
                                         torch.abs(X["jet0_eta"] - X["SDVSecVtx_L_eta"]),
                                         ), dim=0).permute(1,0)[..., np.newaxis]
    
    out_dict["label"] = X['SDVSecVtx_matchedLLPnDau_bydau'].permute(1,0)
    
    torch.nan_to_num(out_dict["tk_pair_features"], nan=0., out=out_dict["tk_pair_features"])

    # torch.nan_to_num(out_dict["tk_features"][:,[4,6],:], nan=0.,  out=out_dict["tk_features"][:,[4,6],:])
    # torch.nan_to_num(out_dict["tk_features"],      nan=-1., out=out_dict["tk_features"])
    torch.nan_to_num(out_dict["tk_features"],      nan=0., out=out_dict["tk_features"])

    # torch.nan_to_num(out_dict["sv_features"][:,4,:], nan=-5.,  out=out_dict["sv_features"][:,4,:])
    # torch.nan_to_num(out_dict["sv_features"],        nan=-1., out=out_dict["sv_features"])
    torch.nan_to_num(out_dict["sv_features"],        nan=0., out=out_dict["sv_features"])



    return out_dict