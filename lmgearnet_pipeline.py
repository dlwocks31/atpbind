from lib.pipeline import Pipeline
import pandas as pd

GPU = 0

def run_exp_pure(lm_layer, lm_train_layer, patience, batch_size=8):
    if lm_layer not in [33, 36]:
        raise ValueError('lm_layer must be 33 or 36')
    lm_type = f'esm-t{lm_layer}'
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
        batch_size=batch_size,
    )
    
    pipeline.model.freeze_lm(
        freeze_all=lm_train_layer == 0,
        freeze_layer_count=lm_layer - lm_train_layer
    )

    train_record = pipeline.train_until_fit(patience=patience)
    return train_record


def read_initial_csv(path):
    try:
        return pd.read_csv(path)
    except (FileNotFoundError, IndexError):
        # File does not exist, or it is empty
        return pd.DataFrame()

def run_parameter_and_save(parameters):
    CSV_PATH = 'lmgearnet_pipeline.csv'
    
    patience = parameters['patience']
    print(pd.Timestamp.now(), parameters)
    result = run_exp_pure(**parameters)
    new_row = pd.DataFrame.from_dict([{**parameters, **result[-1-patience], 'epoch_count': len(result)}])
    df = read_initial_csv(CSV_PATH)
    df = pd.concat([df, new_row])
    df.to_csv(CSV_PATH, index=False)

def main():
    CSV_PATH = 'lmgearnet_pipeline.csv'
    df = read_initial_csv(CSV_PATH)

    for i in range(10):
        for batch_size in [1, 2, 4, 8]:
            run_parameter_and_save({
                "lm_layer": 33,
                "lm_train_layer": 3,
                "patience": 5,
                "batch_size": batch_size,
            })

if __name__ == '__main__':
    main()
