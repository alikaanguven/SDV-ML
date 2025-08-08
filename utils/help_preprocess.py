import numpy as np
import awkward as ak
import torch
from numba import njit
import math

def pad_and_fill(X, branchDict, svDim=12, tkDim=10, fillValue=float("nan")):
    def flatten_and_tensor(field, broadcast_to=None):
        ak_arr = X[field]
        if broadcast_to == 'sv':
            # Broadcast to any sv branch shape
            ak_arr, _ = ak.broadcast_arrays(ak_arr, X[branchDict['sv'][0]])
        else:
            pass
        flat = ak.flatten(ak_arr, axis=1)
        X_np = flat.to_numpy()
        return torch.tensor(X_np)

    X_dict = {}
    for field in X.fields:
        if branchDict['ev'] and field in branchDict['ev']:
            if field.startswith('n'):
                X[field] = ak.values_astype(X[field], np.int32)
            
            X_dict[field] = flatten_and_tensor(field, broadcast_to='sv')
        
        elif branchDict['sv'] and field in branchDict['sv']:
            X_dict[field] = flatten_and_tensor(field)
        elif branchDict['label'] and field in branchDict['label']:
            X_dict[field] = flatten_and_tensor(field)
        elif branchDict['lut'] and field == "SDVIdxLUT_TrackWeight":
            trIdx = X.SDVIdxLUT_TrackIdx
            svIdx = X.SDVIdxLUT_SecVtxIdx
            n_sv =  X.nSDVSecVtx
            
            builder = ak.ArrayBuilder()
            deepTable_weight(X[field], svIdx, n_sv, builder)
            deepX = builder.snapshot()

            almost_flat = ak.flatten(deepX, axis=1)
            padded = ak.pad_none(almost_flat, target=tkDim, clip=True, axis=1)
            filled = ak.fill_none(padded, 0., axis=1)

            X_np = filled.to_numpy()
            X_dict[field] = torch.tensor(X_np)
        elif branchDict['tk'] and field in branchDict['tk']:
            trIdx = X.SDVIdxLUT_TrackIdx
            svIdx = X.SDVIdxLUT_SecVtxIdx
            n_sv =  X.nSDVSecVtx
            
            builder = ak.ArrayBuilder()
            deepTable(X[field], trIdx, svIdx, n_sv, builder)
            deepX = builder.snapshot()
            # print(deepX.type)
            # print('deepX: ', deepX)

            field_fillValue = {
                'SDVTrack_E': 1e3,
                'SDVTrack_pz': 0
            }.get(field, fillValue)         # Returns 'fillValue' if the 'field' does not exist.

            almost_flat = ak.flatten(deepX, axis=1)
            # print('almost_flat: ', almost_flat)
            padded = ak.pad_none(almost_flat, target=tkDim, clip=True, axis=1)
            filled = ak.fill_none(padded, field_fillValue, axis=1)

            X_np = filled.to_numpy()
            X_dict[field] = torch.tensor(X_np)

    return X_dict


@njit
def deepTable(tkBranch, trIdx, svIdx, n_sv, builder):
    """
    Takes the track level branch and converts its shape
    from: (nEvent * var * float32) to (nEvent * var * var * float32)
    representing the association of each tk with sv.

    svIdx: [[0, 0, 1, 1,  2, 2,  3,  3, ...], ...]
    trIdx: [[1, 3, 3, 28, 7, 8, 10, 11, ...], ...]
    n_sv:  [4, 0, 6, 1, 1, 7, ...]
    """
    # at event level depth already
    # every element added are at event level depth
    for ev in range(len(n_sv)):                   # 
        builder.begin_list()                      # adding sv level depth
        for sv in range(n_sv[ev]):                # for each vtx ... 
            builder.begin_list()
            for i2, col in enumerate(svIdx[ev]):  # getting an svIdx from the svIdxs for that event
                if col == sv:                     # if the same svIdx 
                    builder.append(tkBranch[ev][trIdx[ev][i2]])
            builder.end_list()
        builder.end_list()

@njit
def deepTable_weight(wgtBranch, svIdx, n_sv, builder):
    for ev in range(len(n_sv)):
        builder.begin_list()
        for sv in range(n_sv[ev]):
            builder.begin_list()
            for i2, col in enumerate(svIdx[ev]):     # same walk through LUT
                if col == sv:
                    builder.append(wgtBranch[ev][i2])  # ← no trIdx here
            builder.end_list()
        builder.end_list()



# def delta_phi_abs(phi1, phi2):
#     """
#     |Δφ| computed as arccos(cos(φ₁ - φ₂)).
#     Works on Awkward arrays / NumPy scalars alike.
#     Result is in [0, π].
#     """
#     return np.arccos(np.cos(phi1 - phi2))


def calc_dR_deta_dphi(v_phi, v_eta, j_phi, j_eta):
    builder = ak.ArrayBuilder()
    n_ev =  ak.num(v_phi, axis=0)
    n_vtx = ak.num(v_phi, axis=1)
    n_jet = ak.num(j_phi, axis=1)
    vertex_min_dR_deta_dphi(v_phi, v_eta,
                            j_phi, j_eta,
                            int(n_ev), n_vtx, n_jet, builder)
    arrs = builder.snapshot()
    best_dR   = arrs[...,0]
    best_dphi = arrs[...,1]
    best_deta = arrs[...,2]
    return (best_dR, best_dphi, best_deta)

@njit
def vertex_min_dR_deta_dphi(v_phi, v_eta, j_phi, j_eta, n_ev, n_vtx, n_jet, builder):
    for ev in range(n_ev):
        nv = n_vtx[ev]
        nj = n_jet[ev]
        builder.begin_list()
        for vtx in range(nv):
            builder.begin_list()
            best_dR   = None
            best_dphi = None
            best_deta = None
            for jet in range(nj):
                dphi = math.acos(math.cos(j_phi[ev][jet] - v_phi[ev][vtx]))
                deta = abs(j_eta[ev][jet] - v_eta[ev][vtx])
                dR   = math.hypot(dphi, deta)
                if (best_dR is None) or (dR < best_dR):
                    best_dR = dR
                    best_dphi = best_dphi
                    best_deta = best_deta

            builder.append(best_dR)
            builder.append(best_dphi)
            builder.append(best_deta)
            builder.end_list()                     # ── end vertex list
        builder.end_list()                         # ── end event list


# def vertex_min_dR_deta_dphi(v_phi, v_eta, j_phi, j_eta):
#     """
#     """
# 
#     # broadcast vertices against jets  →  (event, vertex, jet)
#     j_phi3 = j_phi[:, None, :]             # (E, 1, J)
#     j_eta3 = j_eta[:, None, :]
#     v_phi3 = v_phi[:, :, None]             # (E, V, 1)
#     v_eta3 = v_eta[:, :, None]
#     
# 
#     # Δφ, Δη, ΔR
#     dphi = delta_phi_abs(j_phi3, v_phi3)   # |Δφ|  (E,V,J)
#     deta = np.abs(j_eta3 - v_eta3)         # |Δη|
#     dR   = np.sqrt(dphi**2 + deta**2)      # ΔR
# 
#     # nearest-jet index for every vertex
#     idx = ak.argmin(dR, axis=-1)
#     print('dR: ', dR.type)
#     print('idx: ', idx.type)
#     min_dR = dR[idx[:,None,:]]
# 
#     min_dR, min_dphi, min_deta = map(gather, (dR, dphi, deta))
#     print('min_dR: ', min_dR.type)
#     exit()
#     return (min_dR[...,0], min_dphi[...,0], min_deta[...,0])