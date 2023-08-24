from lib.pipeline import Pipeline
import pandas as pd

DATA_FILE_PATH = 'lmgearnet_pipeline.json'
GPU = 0

def run_exp_pure(lm_train_layer, patience):
    lm_type, total_layer = 'esm-t33', 33
    pipeline = Pipeline(
        model='lm-gearnet',
        dataset='atpbind3d',
        gpus=[GPU],
        model_kwargs={
            'gpu': GPU,
            'lm_type': lm_type,
            'gearnet_hidden_dim_size': 512,
            'gearnet_hidden_dim_count': 4,
        },
        batch_size=8,
    )
    
    pipeline.model.freeze_lm(
        freeze_all=lm_train_layer == 0,
        freeze_layer_count=total_layer - lm_train_layer
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
    CSV_PATH = 'lmgearnet_pipeline.csv'
    df = read_initial_csv(CSV_PATH)

    for trial in range(10):
        for train_layer in [1, 2, 3]:
            patience = 5
            parameters = {
                "lm_train_layer": train_layer,
                "patience": patience,
            }
            print(pd.Timestamp.now(), parameters)
            result = run_exp_pure(**parameters)
            new_row = pd.DataFrame.from_dict([{**parameters, **result[-1-patience], 'epoch_count': len(result)}])
            df = pd.concat([df, new_row])
            df.to_csv(CSV_PATH, index=False)

if __name__ == '__main__':
    main()
