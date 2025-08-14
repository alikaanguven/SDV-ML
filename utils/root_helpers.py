import ROOT
import os



def clone_without_branch(src_path, dst_path, target_tree, bad_branch):
    """
    Copy every object from `src_path` to `dst_path`, **except**:
      - in the TTree named `target_tree`, the branch `bad_branch` is disabled  
      - all other TTrees keep every branch
    
    Parameters
    ----------
    src_path   : str or pathlib.Path - existing ROOT file
    dst_path   : str or pathlib.Path - file to create/overwrite
    target_tree: str                 - name of the TTree that holds the branch to ignore
    bad_branch : str                 - name of the branch to ignore in `target_tree`
    """
    src_path, dst_path = map(str, (src_path, dst_path))          # accept Path objects
    fin  = ROOT.TFile.Open(src_path, "READ")
    fout = ROOT.TFile.Open(dst_path, "RECREATE")

    for key in fin.GetListOfKeys():
        obj = key.ReadObj()

        # (a) plain histograms, canvases, etc. - write as-is
        if not obj.InheritsFrom("TTree"):
            fout.cd()
            obj.Write(key.GetName())
            continue

        # (b) TTrees - decide branch mask
        tree = obj
        tree.SetBranchStatus("*", True)          # start with everything on
        if tree.GetName() == target_tree:
            tree.SetBranchStatus(bad_branch, False)

        fout.cd()
        newtree = tree.CloneTree(-1, "fast")     # obey current branch mask
        newtree.Write()

        # important: restore default mask for safety
        tree.SetBranchStatus("*", True)


    fout.Close()
    fin.Close()
    print(f"✓ Wrote '{dst_path}' with branch '{bad_branch}' removed from tree '{target_tree}'")

def checkIfCorrupted(file):
    f = ROOT.TFile(file, 'READ')
    if f.IsZombie():
        print(f'{file} could not processed.')
        raise RuntimeError(f'{file} is zombie.')
    else:
        try:
            tree = f.Get("Events")
            _ = tree.GetEntries()
        except Exception as e:
            print('file: ', file)
            print('Exception: \n', e.message, e.args)
            raise RuntimeError(f'{file} is possibly corrupted.')
    f.Close()

def checkIfBranchExists(tree, branchname):
    returnFlag=False
    for branch in tree.GetListOfBranches():
        if ( branch.GetName() == branchname ):
            returnFlag = True
            break
        else:
            returnFlag = False
    return returnFlag

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