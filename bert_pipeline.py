import json
from lib.pipeline import Pipeline
from lib.disable_logger import DisableLogger

DATA_FILE_PATH = 'bert_pipeline.json'

GPU = 1

def exp_record_exists(data, parameters):
    for record in data:
        if all(record[k] == v for k, v in parameters.items()):
            return True
    return False

def run_exp(data, train_layer, trial):
    parameters = {
        "train_layer": train_layer,
        "trial": trial,
    }
    # check if the experiment has been run
    if exp_record_exists(data, parameters):
        print(
            f'Experiment {parameters} has been run, skip')
        return

    pipeline = Pipeline(
        model='bert',
        dataset='atpbind',
        gpus=[GPU],
        model_kwargs={
            'freeze_bert': train_layer == 0,
            'freeze_layer_count': 30 - train_layer,
        })

    for epoch in range(5):
        with DisableLogger():
            pipeline.train(num_epoch=1)
            data.append(parameters | {
                "epoch": epoch,
                "data": pipeline.evaluate(),
            })
        print(data[-1])

    data.sort(key=lambda x: (x["train_layer"], x["trial"], x["epoch"]))
    with open(DATA_FILE_PATH, 'w') as f:
        json.dump(data, f)


def main():
    try:
        with open(DATA_FILE_PATH, 'r') as f:
            data = json.load(f)
    except (FileNotFoundError, IndexError):
        # File does not exist, or it is empty
        data = []

    for trial in range(3):
        for train_layer in range(0, 5):
            run_exp(data, train_layer, trial)


if __name__ == '__main__':
    main()
