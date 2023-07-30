import json
from lib.pipeline import Pipeline
from lib.disable_logger import DisableLogger
import torch
import pandas as pd

GPU = 0

# run experiment without storing data
def run_exp_pure(bert_freeze_layer, 
                 pretrained_layers,
                 bce_weight=1.0,
                 gearnet_freeze_layer=0,
                 reg_weight=0,
                 graph_sequential_max_distance=2,
                 pretrained_weight_file='ResidueType_lmg_4_512_0.57268.pth',
                 pretrained_weight_has_lm=False,
                 ):
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
        optimizer_kwargs={
            'weight_decay': reg_weight,
        },
        bce_weight=bce_weight,
        graph_sequential_max_distance=graph_sequential_max_distance,
        batch_size=4,
    )
    state_dict = torch.load(pretrained_weight_file, map_location=f'cuda:{GPU}')
    if pretrained_weight_has_lm: # LM is also pretrained
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


def main_bce_weight():
    FILE_PATH = 'lmg_pretrained_pipeline_v2.json'
    data = read_initial_data(FILE_PATH)
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
            with open(FILE_PATH, 'w') as f:
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
    PRETRAINED_WEIGHT_FILE = 'lmg_29_4_512_0.58736.pth'
    data = read_initial_data(FILE_PATH)
    for trial in range(5):
        parameters = {
            "gearnet_freeze_layer": 0,
            "bert_freeze_layer": 29,
            "pretrained_layers": 4,
            "bce_weight": 2,
        }
        print(parameters)
        result = run_exp_pure(
            **parameters,
            pretrained_weight_file=PRETRAINED_WEIGHT_FILE,
            pretrained_weight_has_lm=True
        )
        data = add_to_data(data, parameters, result)
        with open(FILE_PATH, 'w') as f:
            json.dump(data, f, indent=2)

def main_lmg_30_4_6():
    CSV_PATH = 'lmg_pretrained_pipeline_30_4_6.csv'
    PRETRAINED_WEIGHT_FILE = 'lmg_30_4_6_0.59149.pth'
    df = read_initial_csv(CSV_PATH)
    for trial in range(5):
        parameters = {
            "gearnet_freeze_layer": 0,
            "bert_freeze_layer": 29, # pretrain할때는 freeze 30, finetune할때는 freeze 29
            "pretrained_layers": 4,
            "bce_weight": 2,
            "graph_sequential_max_distance": 6, # test point
        }
        print(parameters)
        result = run_exp_pure(
            **parameters,
            pretrained_weight_file=PRETRAINED_WEIGHT_FILE,
            pretrained_weight_has_lm=True
        )
        
        new_row = pd.DataFrame.from_dict([{**parameters, **result[-4]}])
        df = pd.concat([df, new_row])
        df.to_csv(CSV_PATH, index=False)

def read_initial_data(path):
    try:
        with open(path, 'r') as f:
            data = json.load(f)
    except (FileNotFoundError, IndexError):
        # File does not exist, or it is empty
        data = []
    return data

def read_initial_csv(path):
    try:
        return pd.read_csv(path)
    except (FileNotFoundError, IndexError):
        # File does not exist, or it is empty
        return pd.DataFrame()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Script options")
    parser.add_argument("--gpu", type=int, default=0,
                        help="GPU to use")
    parser.add_argument("--fn", type=str, required=True,
                        help="Function to execute")
    args = parser.parse_args()

    GPU = args.gpu
    globals()[args.fn]()
