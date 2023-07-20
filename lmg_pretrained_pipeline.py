import json
from lib.pipeline import Pipeline
from lib.disable_logger import DisableLogger
import torch

PRETRAINED_WEIGHT = {
    3: 'ResidueType_lmg_3_512_0.54631.pth',
    4: 'ResidueType_lmg_4_512_0.57268.pth',
    6: 'ResidueType_lmg_6_512_0.55843.pth',
    '29_4': 'lmg_29_4_512_0.58736.pth',
}
GPU = 0

# run experiment without storing data
def run_exp_pure(bert_freeze_layer, pretrained_layers, bce_weight=1.0, gearnet_freeze_layer=0, reg_weight=0):
    pipeline = Pipeline(
        model='lm-gearnet',
        dataset='atpbind3d',
        gpus=[GPU],
        model_kwargs={
            'gpu': GPU,
            'gearnet_hidden_dim_size': 512,
            'gearnet_hidden_dim_count': pretrained_layers if isinstance(pretrained_layers, int) else int(pretrained_layers.split('_')[1]),
            'bert_freeze': bert_freeze_layer == 30,
            'bert_freeze_layer_count': bert_freeze_layer,
        },
        optimizer_kwargs={
            'weight_decay': reg_weight,
        },
        bce_weight=bce_weight,
        batch_size=8,
    )
        
    state_dict = torch.load(PRETRAINED_WEIGHT[pretrained_layers], map_location=f'cuda:{GPU}')
    if isinstance(pretrained_layers, str): # LM is also pretrained
        pipeline.model.load_state_dict(state_dict)
    else:
        pipeline.model.gearnet.load_state_dict(state_dict)
        pipeline.model.freeze_gearnet(freeze_layer_count=gearnet_freeze_layer)

    train_record = pipeline.train_until_fit(patience=3)
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


def main_bce_weight(data, file_path):
    for trial in range(3):
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
            with open(file_path, 'w') as f:
                json.dump(data, f, indent=2)

def main_l2_reg(data, file_path):
    for trial in range(5):
        bert_freeze_layer = 29
        pretrained_layers = 4
        reg_weights = [0, 2e-5, 5e-5, 1e-4, 2e-4]
        for weight in reg_weights:
            parameters = {
                "gearnet_freeze_layer": 0,
                "bert_freeze_layer": bert_freeze_layer,
                "pretrained_layers": pretrained_layers,
                "reg_weight": weight,
            }
            print(parameters)
            result = run_exp_pure(**parameters)
            data = add_to_data(data, parameters, result)
            with open(file_path, 'w') as f:
                json.dump(data, f, indent=2)

def main_lmg_29_4():
    FILE_PATH = 'lmg_pretrained_pipeline_29_4.json'
    data = read_initial_data(FILE_PATH)
    for trial in range(5):
        bert_freeze_layer = 29
        pretrained_layers = 4
        parameters = {
            "gearnet_freeze_layer": 0,
            "bert_freeze_layer": bert_freeze_layer,
            "pretrained_layers": pretrained_layers,
            "bce_weight": 2,
        }
        print(parameters)
        result = run_exp_pure(**parameters)
        data = add_to_data(data, parameters, result)
        with open(FILE_PATH, 'w') as f:
            json.dump(data, f, indent=2)

def read_initial_data(path):
    try:
        with open(path, 'r') as f:
            data = json.load(f)
    except (FileNotFoundError, IndexError):
        # File does not exist, or it is empty
        data = []
    return data

if __name__ == '__main__':
    main_lmg_29_4()
