import sys
import os

sys.path.insert(0, os.path.abspath('..'))

import pandas as pd
from lib.pipeline import Pipeline
import torch


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
                 undersample_rate=0.1,
                 early_stop_metric='valid_mcc',
                 optimizer='adam',
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
            'weight_decay': 0.0001 if optimizer == 'adamw' else 0,
        },
        task_kwargs={
            'use_rus': use_rus,
            'rus_seed': rus_seed,
            'undersample_rate': undersample_rate,
        },
        bce_weight=bce_weight,
        batch_size=batch_size,
        optimizer=optimizer,
    )
    if pretrained_weight_file is not None:
        state_dict = torch.load(pretrained_weight_file,
                                map_location=f'cuda:{GPU}')
        if pretrained_weight_has_lm:
            pipeline.model.load_state_dict(state_dict)
        else:
            pipeline.model.gearnet.load_state_dict(state_dict)

    pipeline.model.freeze_gearnet(freeze_layer_count=gearnet_freeze_layer)

    train_record, state_dict = pipeline.train_until_fit(
        patience=patience, 
        early_stop_metric=early_stop_metric,
        return_state_dict=True
    )
    return (train_record, state_dict)


def read_initial_csv(path):
    try:
        return pd.read_csv(path)
    except (FileNotFoundError, IndexError):
        # File does not exist, or it is empty
        return pd.DataFrame()


def main(undersample_rate, seed_start, seed_end, lr):
    CSV_PATH = 'rus_pipeline.csv'
    
    for trial in range(3):
        for seed in range(seed_start, seed_end):
            print(f'Start {trial} at {pd.Timestamp.now()}, seed={seed}')
            patience = 10
            parameters = {
                "bce_weight": 1,
                "bert_freeze_layer": 27,
                "gearnet_freeze_layer": 0,
                "lr": lr,
                "patience": patience,
                "use_rus": True,
                "rus_seed": seed,
                "undersample_rate": undersample_rate,
                "early_stop_metric": "valid_mcc",
                "optimizer": "adam",
            }
            print(parameters)
            result, state_dict = run_exp_pure(
                **parameters,
                pretrained_layers=4,
                pretrained_weight_file='../ResidueType_lmg_4_512_0.57268.pth',
                pretrained_weight_has_lm=False,
            )
            max_valid_mcc_row = result[-1-patience]
            new_row = pd.DataFrame.from_dict(
                [{**parameters, **max_valid_mcc_row}])
            df = read_initial_csv(CSV_PATH)
            df = pd.concat([df, new_row])
            df.to_csv(CSV_PATH, index=False)
            # save model
            early_stop_metric_name = parameters['early_stop_metric']

            prefix = f'rus_{int(undersample_rate*100)}_{seed}'
            if early_stop_metric_name == 'valid_bce':
                prefix += '_bce'
            
            early_stop_metric = max_valid_mcc_row[early_stop_metric_name]
            if should_save(prefix, early_stop_metric, lower_is_better=(early_stop_metric_name == 'valid_bce')):
                files = find_files_with_prefix(prefix)
                print(
                    f'files: {files}, {early_stop_metric_name}: {early_stop_metric}')
                encoder_layers = [f'model.bert_model.encoder.layer.{i}' for i in range(parameters['bert_freeze_layer'], 30)]
                torch.save({
                    k: v for k, v in state_dict.items()
                    if (not k.startswith('model.bert_model.encoder.layer') or
                        any(k.startswith(layer) for layer in encoder_layers)
                        )
                },
                    f'{prefix}_{early_stop_metric:.4f}.pth'
                )


def find_files_with_prefix(prefix):
    import re
    current_dir = os.getcwd()
    files_with_prefix = []
    for filename in os.listdir(current_dir):
        match_obj = re.match(f'{prefix}_(0\.\d*)\.pth', filename)
        if match_obj:
            files_with_prefix.append((filename, float(match_obj.group(1))))
    return files_with_prefix


def should_save(prefix, current_metric, lower_is_better=False):
    files_with_prefix = find_files_with_prefix(prefix)
    files_with_prefix.sort(key=lambda x: x[1])
    if len(files_with_prefix) == 0:
        return True
    return current_metric < files_with_prefix[0][1] if lower_is_better else current_metric > files_with_prefix[0][1]


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--undersample_rate', type=float, default=0.05)
    parser.add_argument('--seed_start', type=int, default=0)
    parser.add_argument('--seed_end', type=int, default=10)
    parser.add_argument('--lr', type=float, default=2e-4)
    args = parser.parse_args()
    GPU = args.gpu
    main(undersample_rate=args.undersample_rate, 
         seed_start=args.seed_start, 
         seed_end=args.seed_end,
         lr=args.lr
         )
