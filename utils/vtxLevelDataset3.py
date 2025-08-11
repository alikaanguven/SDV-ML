import random
import copy

import numpy as np
import awkward as ak
import uproot
import torch

from numba import njit


def superbatch_iterator(files, keys, superbatch_size=1024*100):
    """
    Reduces the overhead caused by reading data in small batches.
    Now the batch size and data load size are independent.
    WARNING: data load size should be a multiple of batch size.

    Parameters
    ----------
    files: file paths as in uproot.iterator
    keys: TBranch names in list
    superbatch_size: batch_size * multiplet 
    """
    arrays = []
    num_entries = 0
    iterator = uproot.iterate(files, keys, step_size=superbatch_size, library="ak", 
                             #  decompression_executor=ThreadPoolExecutor(max_workers=2) # -- parallel xz/zstd)
                              ) 

    for it in iterator:
        it_length = it.type.length

        if num_entries + it_length < superbatch_size:
            # Batch is missing some entries, get more entries
            arrays.append(it)
            num_entries += it_length
        else:
            # If batch is ready or more than ready, handle it
            remaining = superbatch_size - num_entries

            if remaining > 0:
                arrays.append(it[:remaining])
                yield ak.concatenate(arrays)  # next iteration starts directly after this line.
                arrays = [it[remaining:]]
                num_entries = it_length - remaining
            else:
                yield ak.concatenate(arrays)  # next iteration starts directly after this line.
                arrays = []
                num_entries = 0
                # arrays = [it]
                # num_entries = it_length

    if arrays:
        yield ak.concatenate(arrays)



class ModifiedUprootIterator(torch.utils.data.IterableDataset):
    def __init__(self, files, branches, shuffle=False, nWorkers=1, step_size=100):
        """
        Parameters
        ----------
        files : dict
                keys: "sig", "bkg"
                values: ['path_to_file:Events', ...]
        branches : dict
                dict of branches in TTree.
                keys should be ev, sv, tk.
                values should be of type list.
        nWorkers : int
                Files will be divided among the workers.
                Therefore, nWorkers determines number of divisions.
                nWorkers=0 will be treated as nWorkers=1.
        step_size : int
                number of Events to be read from the files at each iteration.
        """
        
        print('Initialize iterable dataset')
        self.files = files
        self.branches = branches

        self.branchList = [b for key, value in branches.items() if value is not None for b in value]
        
        self.step_size = step_size
        self.nWorkers = max(nWorkers, 1)
        self.shuffle = shuffle
        print('nWorkers: ', self.nWorkers)
        
        if self.shuffle:
            random.shuffle(self.files['sig'])
            if self.files['bkg']:
                random.shuffle(self.files['bkg'])

        if self.files['bkg']:
            self.workerBkgList = self._distribute_files(self.files['bkg'])
        else:
            self.workerBkgList = None
        self.workerSigList = self._copy_files(self.files['sig'], shuffle=self.shuffle)

        self.SigIteratorList = None
        self.BkgIteratorList = None
        self._refresh_iterators()
        
        self.x = None
        self.xSig = None
        self.xBkg = None

    
    def _distribute_files(self, files):
        return [[files[i] for i in range(len(files)) if i % self.nWorkers == worker_id] for worker_id in range(self.nWorkers)]

    def _copy_files(self, files, shuffle=True):
        workerList = []
        for worker_info_id in range(self.nWorkers):
            files_copy = copy.copy(files)
            if shuffle: random.shuffle(files_copy)
            workerList.append(files_copy)
        return workerList
        
    def _refresh_iterators(self, shuffle=False):
        self.workerSigList = self._copy_files(self.files['sig'], shuffle=shuffle)
        self.SigIteratorList = [superbatch_iterator(workerFiles, self.branchList, superbatch_size=self.step_size) for workerFiles in self.workerSigList]
        if self.workerBkgList:
            self.BkgIteratorList = [superbatch_iterator(workerFiles, self.branchList, superbatch_size=self.step_size) for workerFiles in self.workerBkgList]
        else:
            self.BkgIteratorList = None

    def __iter__(self):
        print('__iter__ is called.')
        if self.step_size <200: 
            self.step_size += 25
            print('step_size is increased to ', self.step_size)
        self._refresh_iterators(shuffle=self.shuffle)
        return self

    def update_step_size(self, new_step_size):
        self.step_size = new_step_size

    
    def __next__(self):
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info else 0

        if self.BkgIteratorList:
            self.xBkg = next(self.BkgIteratorList[worker_id])
            try:
                self.xSig = next(self.SigIteratorList[worker_id])
            except StopIteration:
                print(f'Worker {worker_id}s SigIteratorList is exhausted. Loading again.')
                self.SigIteratorList[worker_id] = superbatch_iterator(self.workerSigList[worker_id], self.branchList, superbatch_size=self.step_size)
                self.xSig = next(self.SigIteratorList[worker_id])
            self.x = ak.concatenate([self.xBkg, self.xSig])
        else:
            self.xSig = next(self.SigIteratorList[worker_id])
            self.x = self.xSig

        if self.shuffle:
            self.x = self._shuffle_akArr(self.x)
        self._add_four_vector_branches()
        return self.x


    def _shuffle_akArr(self, x):
        """ Shuffle awkward array. """
        idx = np.arange(len(x))
        np.random.shuffle(idx)
        return x[idx]

    
    def _add_four_vector_branches(self):
        if all(x in self.branchList for x in ['SDVTrack_pt', 'SDVTrack_eta', 'SDVTrack_phi']) and \
           any(x not in self.branchList for x in ['SDVTrack_E', 'SDVTrack_px', 'SDVTrack_py', 'SDVTrack_pz']):
            
            self.branches['tk'].extend(['SDVTrack_E', 'SDVTrack_px', 'SDVTrack_py', 'SDVTrack_pz'])
            E, px, py, pz = ptetaphim_to_epxpypz(self.x['SDVTrack_pt'], self.x['SDVTrack_eta'], self.x['SDVTrack_phi'])
            self.x['SDVTrack_E'] = E
            self.x['SDVTrack_px'] = px
            self.x['SDVTrack_py'] = py
            self.x['SDVTrack_pz'] = pz

def ptetaphim_to_epxpypz(pt, eta, phi, m=0.13957):
    px = pt * np.cos(phi)
    py = pt * np.sin(phi)
    pz = pt * np.sinh(eta)
    E = np.sqrt(px*px + py*py + pz*pz + m*m)
    return (E, px, py, pz)




def stable_iterator(files, keys, superbatch_size=100, drop_last=False):
    """
    Retrieves same number of entries in each iteration for stable training.

    WARNING: Set drop_last=False in prediction mode,
             otherwise last batch will be dropped!!

    Parameters
    ----------
    files: file paths as in uproot.iterator
    keys: TBranch names in list
    superbatch_size: batch_size * multiplet 
    """
    arrays = []
    num_entries = 0
    iterator = uproot.iterate(files, keys, step_size=superbatch_size, library="ak",
                              ) 

    for it in iterator:
        it_length = it.type.length

        if num_entries + it_length < superbatch_size:
            # Batch is missing some entries, get more entries
            arrays.append(it)
            num_entries += it_length
        else:
            # If batch is ready or more than ready, handle it
            remaining = superbatch_size - num_entries

            if remaining > 0:
                arrays.append(it[:remaining])
                yield ak.concatenate(arrays)  # next iteration starts directly after this line.
                arrays = [it[remaining:]]
                num_entries = it_length - remaining
            elif remaining == 0:
                yield ak.concatenate(arrays)  # next iteration starts directly after this line.
                arrays = []
                num_entries = 0
            else:
                raise RuntimeError("This basically should not happen.")


    # Sometimes last batch has only 1 element which is not enough for BatchNorm layers.
    # They fail the training quite unexpectedly, after many epochs.
    # Handle the last batch
    if arrays and not drop_last:
        yield ak.concatenate(arrays)
    else:
        pass # Last batch is dropped