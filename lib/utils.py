import torch
from statistics import mean, stdev

def dict_tensor_to_num(d):
    return {k: v.item() if isinstance(v, torch.Tensor) else v
             for k, v in d.items()}
    
def round_dict(d, n):
    return {k: round(v, n) if isinstance(v, float) else v
                for k, v in d.items()}
    
def statistics_per_key(list_of_dict):
    keys = list_of_dict[0].keys()
    result = {}
    for key in keys:
        result[key] = [mean([i[key] for i in list_of_dict]), stdev([i[key] for i in list_of_dict]) if len(list_of_dict) >= 2 else -1, len(list_of_dict)]
    return result