"""
Contains code for group weight orthogonalization via regularization. Both inter-group and intra-group.
"""
from typing import List
import torch
import re

GOR_REG_TYPES = ['inter', 'intra']


def calc_dist(w: torch.tensor):

    n_rows, n_cols = w.shape[1:]
    if n_rows >= n_cols:
        # "Tall" matrix --> ||W.T@W - I||
        # [32,32] matrix (L2 Norm)
        return torch.dist(w.permute(0, 2, 1) @ w, torch.eye(w.shape[2]).cuda()) ** 2
    else:
        # Wide matrix --> ||W@W.T - I||
        return torch.dist(w @ w.permute(0, 2, 1), torch.eye(w.shape[1]).cuda()) ** 2


def intra_reg_loss(w: torch.tensor, group_size: int, num_groups: int):
    """
    loop-less implementation of intra-group orthogonalization
    :param w: weight tensor. Cout x d
    :param group_size: number of filter in each group
    :param num_groups: number of groups within the filter (num_groups * group_size = c_out).
    :return: norm value
    """
    assert w.ndim == 2

    # reshape into a 3d tensor where tensor[i] contains the i'th set to orthogonolize
    # e.g. tensor[0] contains the first filter of every group.
    w_r = w.reshape(num_groups, group_size, -1).permute(1, 0, 2)
    w_f = w_r.reshape(group_size, num_groups, -1)  # group_size x num_groups x d

    # calc distance
    return calc_dist(w_f)


def inter_reg_loss(w: torch.tensor, group_size: int, num_groups: int):
    assert w.ndim == 2
    w_r = w.reshape(num_groups, group_size, -1)  # num_groups x group_size x d
    return calc_dist(w_r)

def check_need_to_regularize(module: torch.nn,
                             name: str,
                             reg_fc: bool,
                             names_to_reg: List[str]) -> bool:
    if isinstance(module, torch.nn.Conv2d) or (reg_fc and isinstance(module, torch.nn.Linear)):
        if module.weight.requires_grad:
            # ----------------------------------------------------------------------------------------------- #
            # only lora layer will be regularize ....
            # Verify that name meets the filtering criterion

            if names_to_reg:
                # only up stream + lora up layer is regularizing ...
                return any([re.search(re_str, name) is not None for re_str in names_to_reg])
            else:
                return True
    return False

def calc_group_reg_loss(model: torch.nn.Module,
                        num_groups: int,
                        reg_type: str,
                        min_num_filters: int = 4,
                        regularize_fc_layers: bool = False,
                        names_to_reg: List[str] = None):
    assert reg_type in GOR_REG_TYPES, f'Unsupported GOR type {reg_type}'
    total_reg_value = 0
    for k, v in model.named_modules():
        if check_need_to_regularize(module=v,name=k,reg_fc=regularize_fc_layers,names_to_reg=names_to_reg):
            # is true = up stream + lora up ...
            c_out = v.weight.shape[0]
            w = v.weight.reshape(c_out, -1)  # flatten to 2D
            actual_num_groups = min(num_groups, c_out // min_num_filters)
            assert c_out % actual_num_groups == 0, f'c_out={c_out} not divisible by {actual_num_groups} groups, ' f'for layer {k}'
            group_size = c_out // actual_num_groups  # Number of filters in each group
            assert group_size > 0, f'Bad group size for {k}. c_out = {c_out}, num_groups = {num_groups}'
            if reg_type == 'intra':
                if group_size == 1:
                    total_reg_value += calc_dist(w.unsqueeze(0))  # calc_dist expects 3d tensor
                else:
                    total_reg_value += intra_reg_loss(w, group_size, actual_num_groups)
            elif reg_type == 'inter':
                total_reg_value += inter_reg_loss(w, group_size, actual_num_groups)
            else:
                raise Exception(f'Unsupported mode {reg_type}')
    return total_reg_value
