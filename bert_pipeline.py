from lib.pipeline import Pipeline
from lib.disable_logger import DisableLogger
import pandas as pd

DATA_FILE_PATH = 'bert_pipeline.json'

GPU = 1

def exp_record_exists(data, parameters):
    for record in data:
        if all(record[k] == v for k, v in parameters.items()):
            return True
    return False

def run_exp_pure(train_layer):
    pipeline = Pipeline(
        model='bert',
        dataset='atpbind',
        gpus=[GPU],
        model_kwargs={
            'freeze_bert': train_layer == 0,
            'freeze_layer_count': 30 - train_layer,
        },
        batch_size=8,
        )

    train_record = pipeline.train_until_fit(patience=3)
    return train_record


def read_initial_csv(path):
    try:
        return pd.read_csv(path)
    except (FileNotFoundError, IndexError):
        # File does not exist, or it is empty
        return pd.DataFrame()

def main():
    CSV_PATH = 'bert_pipeline.csv'
    df = read_initial_csv(CSV_PATH)

    for trial in range(3):
        for train_layer in range(0, 5):
            parameters = {
                "train_layer": train_layer,
            }
            print(parameters)
            result = run_exp_pure(**parameters)
            new_row = pd.DataFrame.from_dict([{**parameters, **result[-4]}])
            df = pd.concat([df, new_row])
            df.to_csv(CSV_PATH, index=False)

if __name__ == '__main__':
    main()
