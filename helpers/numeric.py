import numpy as np
import torch


def my_round(ndigits, tensor):
    if torch.is_tensor(tensor):
        return torch.round(tensor * 10 ** ndigits) / (10 ** ndigits)
    else:
        return np.round(tensor, decimals=ndigits)
    return rounded


def tensor_to_list(t):
    return t.detach().numpy().tolist()
