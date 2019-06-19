import miuds.config
import torch


def to_float_tensor(array):
    return torch.Tensor(array).to(miuds.config.device)


def to_tensor(array):
    return torch.from_numpy(array).to(miuds.config.device)
