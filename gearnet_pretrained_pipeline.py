import json
from lib.pipeline import Pipeline
from lib.disable_logger import DisableLogger
import torch

DATA_FILE_PATH = 'gearnet_pretrained_pipeline.json'
TOTAL_LAYER_COUNT = 9
PRETRAINED_WEIGHT = {
    6: 'ResidueType_6_512_0.38472.pth',
    8: 'ResidueType_8_512_0.37701.pth',
    9: 'ResidueType_9_512_0.38827.pth'
}

GPU = 1

def exp_record_exists(data, parameters):
    for record in data:
        if all(record[k] == v for k, v in parameters.items()):
            return True
    return False


def run_exp(data, layer, trial, pretrained_layers):
    parameters = {
        "layer": layer,
        "pretrained_layers": pretrained_layers,
        "trial": trial,
    }
    # check if the experiment has been run
    if exp_record_exists(data, parameters):
        print(f'Experiment {parameters} has been run, skip')
        return
    
    pipeline = Pipeline(
        model='gearnet',
        dataset='atpbind3d',
        gpus=[GPU],
        model_kwargs={
            'input_dim': 21,
            'hidden_dims': [512] * pretrained_layers,
            'gpu': GPU,
        }
     )
    
    pretrained_weight = PRETRAINED_WEIGHT[pretrained_layers]
    
    # load pretrained weight
    state_dict = torch.load(pretrained_weight)
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
            data.append(parameters | {
                "epoch": epoch,
                "data": pipeline.evaluate()
            })
        print(data[-1])
    
    data.sort(key=lambda x: (x["pretrained_layers"], x["layer"], x["trial"], x["epoch"]))
    with open(DATA_FILE_PATH, 'w') as f:
        json.dump(data, f)
    

def main():
    try:
        with open(DATA_FILE_PATH, 'r') as f:
            data = json.load(f)
    except (FileNotFoundError, IndexError):
        # File does not exist, or it is empty
        data = []

    for pretrained_layers in [6, 8, 9]:
        for layer in range(0, pretrained_layers):
            for trial in range(1):
                run_exp(data=data, layer=layer, trial=trial, pretrained_layers=pretrained_layers)

if __name__ == '__main__':
    main()
