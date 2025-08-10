import torch.nn as nn

def parameter_stats(model: nn.Module):
    """
    Return a dictionary with totals for:
      • all parameters  
      • trainable vs. non-trainable  
      • trainable weights vs. trainable biases
    """
    totals = {
        "total": 0,
        "trainable": 0,
        "non_trainable": 0,
        "trainable_weights": 0,
        "trainable_biases": 0,
    }

    for name, param in model.named_parameters():
        n = param.numel()          # number of scalars in the tensor
        totals["total"] += n

        if param.requires_grad:    # counted in the backward pass
            totals["trainable"] += n
            if name.endswith(".weight"):
                totals["trainable_weights"] += n
            elif name.endswith(".bias"):
                totals["trainable_biases"] += n
        else:
            totals["non_trainable"] += n

    return totals


def significance(s,b,b_err):
    """
    Median discovery significance
    Definition at slide 33:
    https://www.pp.rhul.ac.uk/~cowan/stat/cowan_munich16.pdf
    
    """
    return np.sqrt(2*((s+b)*np.log(((s+b)*(b+b_err*b_err))/(b*b+(s+b)*b_err*b_err+1e-20)) - 
                    (b*b/(b_err*b_err + 1e-20))*np.log(1+(b_err*b_err*s)/(b*(b+b_err*b_err)+1e-20))))