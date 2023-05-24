import json
from lib.pipeline import Pipeline
from lib.disable_logger import DisableLogger
import torch

DATA_FILE_PATH = 'gearnet_pretrained_pipeline.json'
TOTAL_LAYER_COUNT = 9
PRETRAINED_WEIGHT = 'ResidueType_9_512_0.38827.pth'
GPU = 2

def run_exp(data, layer, trial):
    # check if the experiment has been run
    if any(d['layer'] == layer and d['trial'] == trial for d in data):
        print(f'Experiment {layer}-{trial} has been run, skip')
        return
    
    pipeline = Pipeline(
        model='gearnet',
        dataset='atpbind3d',
        gpus=[GPU],
        model_kwargs={
            'input_dim': 21,
            'hidden_dims': [512] * 9,
            'gpu': GPU,
        }
     )
    
    # load pretrained weight
    state_dict = torch.load(PRETRAINED_WEIGHT)
    new_state_dict = {}
    for k in state_dict['model'].keys():
        if k.startswith("model"):
            new_state_dict[k.replace("model.", "")] = state_dict['model'][k]

    pipeline.model.load_state_dict(new_state_dict)

    # freeze layers
    pipeline.model.freeze(freeze_layer_count=layer)
    
    for epoch in range(5):
        with DisableLogger():
            pipeline.train(num_epoch=1)
            data.append({"layer": layer, "trial": trial, "epoch": epoch, "data": pipeline.evaluate()})
        print(data[-1])
    
    data.sort(key=lambda x: (x["layer"], x["trial"], x["epoch"]))
    with open(DATA_FILE_PATH, 'w') as f:
        json.dump(data, f)
    


def main():
    try:
        with open(DATA_FILE_PATH, 'r') as f:
            data = json.load(f)
    except (FileNotFoundError, IndexError):
        # File does not exist, or it is empty
        data = []

    for layer in range(0, 9):
        for trial in range(3):
            run_exp(data, layer, trial)

if __name__ == '__main__':
    main()
