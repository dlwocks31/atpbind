import json
from lib.pipeline import Pipeline
from lib.disable_logger import DisableLogger

DATA_FILE_PATH = 'lmgearnet_pipeline.json'
GPU = 2

def exp_record_exists(data, parameters):
    for record in data:
        if all(record[k] == v for k, v in parameters.items()):
            return True
    return False

def run_exp(data, gearnet_hidden_dim_count, bert_unfreeze_layer, trial):
    parameters = {
        "gearnet_hidden_dim_count": gearnet_hidden_dim_count,
        "bert_unfreeze_layer": bert_unfreeze_layer,
        "trial": trial
    }
    # check if the experiment has been run
    if exp_record_exists(data, parameters):
        print(f'Experiment {parameters} has been run, skip')
        return
    
    pipeline = Pipeline(
        model='lm-gearnet',
        dataset='atpbind3d',
        gpus=[GPU],
        model_kwargs={
            'gpu': GPU,
            'gearnet_hidden_dim_size': 512,
            'gearnet_hidden_dim_count': gearnet_hidden_dim_count,
            'bert_freeze': bert_unfreeze_layer == 0,
            'bert_freeze_layer_count': 30 - bert_unfreeze_layer,
    })
    
    for epoch in range(5):
        with DisableLogger():
            pipeline.train(num_epoch=1)
            data.append(parameters | {
                "epoch": epoch,
                "data": pipeline.evaluate()
            })
        print(data[-1])
    
    data.sort(key=lambda x: (x["bert_unfreeze_layer"],
              x["gearnet_hidden_dim_count"], x["trial"], x["epoch"]))
    with open(DATA_FILE_PATH, 'w') as f:
        json.dump(data, f)
    


def main():
    try:
        with open(DATA_FILE_PATH, 'r') as f:
            data = json.load(f)
    except (FileNotFoundError, IndexError):
        # File does not exist, or it is empty
        data = []

    for gearnet_hidden_dim_count in range(3, 8):
        for bert_unfreeze_layer in range(2):
            for trial in range(3):
                run_exp(data, gearnet_hidden_dim_count, bert_unfreeze_layer, trial)

if __name__ == '__main__':
    main()
