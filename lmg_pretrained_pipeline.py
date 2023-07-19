import json
from lib.pipeline import Pipeline
from lib.disable_logger import DisableLogger
import torch

DATA_FILE_PATH = 'lmg_pretrained_pipeline_v2.json'
PRETRAINED_WEIGHT = {
    3: 'ResidueType_lmg_3_512_0.54631.pth',
    4: 'ResidueType_lmg_4_512_0.57268.pth',
    6: 'ResidueType_lmg_6_512_0.55843.pth',   
}
GPU = 0

# run experiment without storing data
def run_exp_pure(gearnet_freeze_layer, bert_freeze_layer, pretrained_layers, bce_weight=1.0):
    pipeline = Pipeline(
        model='lm-gearnet',
        dataset='atpbind3d',
        gpus=[GPU],
        model_kwargs={
            'gpu': GPU,
            'gearnet_hidden_dim_size': 512,
            'gearnet_hidden_dim_count': pretrained_layers,
            'bert_freeze': bert_freeze_layer == 30,
            'bert_freeze_layer_count': bert_freeze_layer,
        },
        bce_weight=bce_weight,
    )
        
    state_dict = torch.load(PRETRAINED_WEIGHT[pretrained_layers], map_location=f'cuda:{GPU}')
    pipeline.model.gearnet.load_state_dict(state_dict)
    pipeline.model.freeze_gearnet(freeze_layer_count=gearnet_freeze_layer)

    train_record = []
    for epoch in range(5):
        with DisableLogger():
            pipeline.train(num_epoch=1)
            train_record.append({
                "epoch": epoch,
                "data": pipeline.evaluate()
            })
        print(train_record[-1])
    return train_record


def add_to_data(file_data, parameters, trial):
    '''
    file_data: list of {parameters: dict, trials: list}
    parameters: dict
    trial: dict

    This function would find the entry in file_data with the same parameters
    and append the trial to the trials list.
    '''
    result = file_data[::]
    for entry in result:
        if all(entry['parameters'][k] == v for k, v in parameters.items()):
            entry['trials'].append(trial)
            return result
    result.append({
        "parameters": parameters,
        "trials": [trial]
    })
    return result


def main():
    try:
        with open(DATA_FILE_PATH, 'r') as f:
            data = json.load(f)
    except (FileNotFoundError, IndexError):
        # File does not exist, or it is empty
        data = []

    for trial in range(5):
        gearnet_freeze_layer = 1
        bert_freeze_layer = 29
        pretrained_layers = 4
        for bce_weight in [4, 2, 1, 0.5, 0.25]:
            parameters = {
                "gearnet_freeze_layer": gearnet_freeze_layer,
                "bert_freeze_layer": bert_freeze_layer,
                "pretrained_layers": pretrained_layers,
                "bce_weight": bce_weight,
            }
            print(parameters)
            result = run_exp_pure(**parameters)
            data = add_to_data(data, parameters, result)
            with open(DATA_FILE_PATH, 'w') as f:
                json.dump(data, f, indent=2)


if __name__ == '__main__':
    main()
