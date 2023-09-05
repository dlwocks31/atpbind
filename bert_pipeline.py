from lib.pipeline import Pipeline
from lib.disable_logger import DisableLogger
import pandas as pd

DATA_FILE_PATH = 'bert_pipeline.json'

GPU = 0

def run_exp_pure(model, train_layer, total_layer, patience=5):
    pipeline = Pipeline(
        model=model,
        dataset='atpbind3d',
        gpus=[GPU],
        model_kwargs={
            'gpu': GPU,
            'freeze_esm': train_layer == 0,
            'freeze_layer_count': total_layer - train_layer,
        },
        batch_size=16,
    )

    train_record = pipeline.train_until_fit(patience=patience)
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

    for trial in range(2):
        for model, total_layer in [('esm-t33', 33)]:
            for train_layer in [3]:
                patience = 5
                parameters = {
                    "total_layer": total_layer,
                    "train_layer": train_layer,
                    "patience": patience,
                }
                print(pd.Timestamp.now(), parameters)
                result = run_exp_pure(model=model, **parameters)
                new_row = pd.DataFrame.from_dict([{**parameters, **result[-1-patience]}])
                df = pd.concat([df, new_row])
                df.to_csv(CSV_PATH, index=False)
                if total_layer == 33:
                    break

if __name__ == '__main__':
    main()
