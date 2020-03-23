import numpy as np
import torch
import torch.nn as nn

def fix_precision(tensor, precision_bits, rm_outlier_frac=100, parameter=True):
    tensor = (tensor * 2**precision_bits).long()
    max_value = max(
        np.abs(np.percentile(tensor, rm_outlier_frac)),
        np.abs(np.percentile(tensor, 100 - rm_outlier_frac))
    )
    cp_tensor = 1 * tensor
    tensor = tensor.clamp(min=-max_value, max=max_value)
    if parameter:
        return nn.Parameter(tensor, requires_grad=False)
    else:
        return tensor
    
def float_precision(tensor, precision_bits):
    tensor = tensor.float()/2**precision_bits
    return tensor