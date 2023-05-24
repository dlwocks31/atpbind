import json
from lib.pipeline import Pipeline
from lib.disable_logger import DisableLogger
import torch

DATA_FILE_PATH = 'lmg_pretrained_pipeline.json'
PRETRAINED_WEIGHT = 'ResidueType_lmg_6_512_0.55843.pth'
GEARNET_HIDDEN_DIM_COUNT = 6
GPU = 3

def exp_record_exists(data, parameters):
    for record in data:
        if all(record[k] == v for k, v in parameters.items()):
            return True
    return False


def run_exp(data, gearnet_freeze_layer, bert_freeze_layer, trial):
    parameters = {
        "gearnet_freeze_layer": gearnet_freeze_layer,
        "bert_freeze_layer": bert_freeze_layer,
        "trial": trial,
    }
    # check if the experiment has been run
    if exp_record_exists(data, parameters):
        print(
            f'Experiment {parameters} has been run, skip')
        return

    pipeline = Pipeline(
        model='lm-gearnet',
        dataset='atpbind3d',
        gpus=[GPU],
        model_kwargs={
            'gpu': GPU,
            'gearnet_hidden_dim_size': 512,
            'gearnet_hidden_dim_count': GEARNET_HIDDEN_DIM_COUNT,
            'bert_freeze': bert_freeze_layer == 30,
            'bert_freeze_layer_count': bert_freeze_layer,
        })
    
    state_dict = torch.load(PRETRAINED_WEIGHT)
    pipeline.model.gearnet.load_state_dict(state_dict)
    pipeline.model.freeze_gearnet(freeze_layer_count=gearnet_freeze_layer)



    for epoch in range(5):
        with DisableLogger():
            pipeline.train(num_epoch=1)
            data.append(parameters | {
                "epoch": epoch,
                "data": pipeline.evaluate()
            })
        print(data[-1])

    data.sort(key=lambda x: (x["gearnet_freeze_layer"],
              x["bert_freeze_layer"], x["trial"], x["epoch"]))
    with open(DATA_FILE_PATH, 'w') as f:
        json.dump(data, f)


def main():
    try:
        with open(DATA_FILE_PATH, 'r') as f:
            data = json.load(f)
    except (FileNotFoundError, IndexError):
        # File does not exist, or it is empty
        data = []

    for gearnet_freeze_layer in range(0, 6):
        for bert_freeze_layer in range(29, 31):
            for trial in range(1):
                run_exp(
                    data = data,
                    gearnet_freeze_layer = gearnet_freeze_layer,
                    bert_freeze_layer = bert_freeze_layer, 
                    trial = trial
                )


if __name__ == '__main__':
    main()
