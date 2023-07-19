import torch

def dict_tensor_to_num(d):
    return {k: v.item() if isinstance(v, torch.Tensor) else v
             for k, v in d.items()}