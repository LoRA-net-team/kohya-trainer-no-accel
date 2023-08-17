import os
import pickle
import torch
from torch import nn

x = torch.FloatTensor([[0, 1, 2], [2, 1, 0]])
x_ = x @ x.T
def cov(tensor, rowvar=True, bias=False):
    """Estimate a covariance matrix (np.cov)"""
    tensor = tensor if rowvar else tensor.transpose(-1, -2)
    tensor = tensor - tensor.mean(dim=-1, keepdim=True)
    factor = 1 / (tensor.shape[-1] - int(not bool(bias)))
    return factor * tensor @ tensor.transpose(-1, -2).conj()

def corrcoef(tensor, rowvar=True):
    """Get Pearson product-moment correlation coefficients (np.corrcoef)"""
    covariance = cov(tensor, rowvar=rowvar)
    variance = covariance.diagonal(0, -1, -2)
    if variance.is_complex():
        variance = variance.real
    stddev = variance.sqrt()
    covariance /= stddev.unsqueeze(-1)
    covariance /= stddev.unsqueeze(-2)
    if covariance.is_complex():
        covariance.real.clip_(-1, 1)
        covariance.imag.clip_(-1, 1)
    else:
        covariance.clip_(-1, 1)
    return covariance

result2 = torch.corrcoef(x)
cov = corrcoef(x, rowvar=True)
print(result2)
print(cov)