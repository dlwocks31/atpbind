import sys, os
sys.path.insert(0, os.path.abspath('..'))
from lib.pipeline import Pipeline
import torch
import pandas as pd

GPU = 0

def run_exp_pure(bert_freeze_layer,
                 pretrained_layers,
                 bce_weight=1.0,
                gearnet_freeze_layer=0,
                pretrained_weight_file='ResidueType_lmg_4_512_0.57268.pth',
                pretrained_weight_has_lm=False,
                lr=1e-3,
                batch_size=4,
                patience=3,
                use_rus=False,
                rus_seed=0,
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
            'lr': lr,
        },
        task_kwargs={
            'use_rus': use_rus,
            'rus_seed': rus_seed,
        },
        bce_weight=bce_weight,
        batch_size=batch_size,
    )
    if pretrained_weight_file is not None:
        state_dict = torch.load(pretrained_weight_file, map_location=f'cuda:{GPU}')
        if pretrained_weight_has_lm:
            pipeline.model.load_state_dict(state_dict)
        else:
            pipeline.model.gearnet.load_state_dict(state_dict)
    
    pipeline.model.freeze_gearnet(freeze_layer_count=gearnet_freeze_layer)

    train_record = pipeline.train_until_fit(patience=patience)
    return (train_record, pipeline)

def read_initial_csv(path):
    try:
        return pd.read_csv(path)
    except (FileNotFoundError, IndexError):
        # File does not exist, or it is empty
        return pd.DataFrame()

def main():
    CSV_PATH = 'rus_pipeline.csv'
    for trial in range(5, 10):
        patience = 3
        parameters = {
            "bce_weight": 1,
            "bert_freeze_layer": 29,
            "gearnet_freeze_layer": 0,
            "lr": 5e-4,
            "patience": patience,
            "use_rus": True,
            "rus_seed": trial,
        }
        print(parameters)
        result, pipeline = run_exp_pure(
            **parameters,
            pretrained_layers=4,
            pretrained_weight_file='../ResidueType_lmg_4_512_0.57268.pth',
            pretrained_weight_has_lm=False,
        )
        new_row = pd.DataFrame.from_dict([{**parameters, **result[-1-patience]}])
        df = read_initial_csv(CSV_PATH)
        df = pd.concat([df, new_row])
        df.to_csv(CSV_PATH, index=False)
        # save model
        torch.save({
            k: v for k, v in pipeline.task.state_dict().items()
                if (not k.startswith('model.bert_model.encoder.layer') or 
                    k.startswith('model.bert_model.encoder.layer.29')
                )
        },
            f'rus_pipeline_{trial}_{result[-1]["mcc"]:.5f}.pth'
        )

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()
    GPU = args.gpu
    main()