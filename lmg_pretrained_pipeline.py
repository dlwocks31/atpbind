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
                 pretrained_weight_file='ResidueType_lmg_4_512_0.57268.pth',
                 pretrained_weight_has_lm=False,
                 lr=1e-3,
                 batch_size=4,
                 knn_k=10,
                 spatial_radius=10.0,
                 sequential_max_distance=2,
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
            'lr': lr,
        },
        bce_weight=bce_weight,
        batch_size=batch_size,
        graph_knn_k=knn_k,
        graph_spatial_radius=spatial_radius,
        graph_sequential_max_distance=sequential_max_distance,
    )
    if pretrained_weight_file is not None:
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
    CSV_PATH = 'lmg_pretrained_pipeline_main_bce_weight.csv'
    df = read_initial_csv(CSV_PATH)
    for trial in range(5):
        for bce_weight in [0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 10.0, 20.0, 40.0]:
            parameters = {
                "bce_weight": bce_weight,
                "lr": 1e-3,
            }
            print(parameters)
            result = run_exp_pure(
                **parameters,
                gearnet_freeze_layer=0,
                bert_freeze_layer=29,
                pretrained_layers=4,
                pretrained_weight_file='ResidueType_lmg_4_512_0.57268.pth',
                pretrained_weight_has_lm=False,
            )
            new_row = pd.DataFrame.from_dict([{**parameters, **result[-4]}])
            df = pd.concat([df, new_row])
            df.to_csv(CSV_PATH, index=False)

def main_edge_type():
    CSV_PATH = 'lmg_pretrained_pipeline_edge_type.csv'
    df = read_initial_csv(CSV_PATH)
    for trial in range(5):
        for knn_k in [10, 20]:
            for spatial_radius in [10.0, 20.0]:
                for sequential_max_distance in range(2, 9):
                    parameters = {
                        'knn_k': knn_k,
                        'spatial_radius': spatial_radius,
                        'sequential_max_distance': sequential_max_distance,
                        'lr': 5e-4,
                    }
                    print(parameters)
                    result = run_exp_pure(
                        **parameters,
                        gearnet_freeze_layer=0,
                        bert_freeze_layer=29,
                        pretrained_layers=4,
                        pretrained_weight_file=None,
                        batch_size=2,
                    )
                    new_row = pd.DataFrame.from_dict([{**parameters, **result[-4]}])
                    df = pd.concat([df, new_row])
                    df.to_csv(CSV_PATH, index=False)
                    
        

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
            "sequential_max_distance": 6, # test point
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
