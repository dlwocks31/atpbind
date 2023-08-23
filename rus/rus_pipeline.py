import sys
import os

sys.path.insert(0, os.path.abspath('..'))

import pandas as pd
from lib.pipeline import Pipeline
import torch
from torchdrug import utils, data
from lib.lr_scheduler import ExponentialLR


GPU = 0
def device():
    return torch.device(f'cuda:{GPU}')


def run_exp_pure(bert_freeze_layer,
                 pretrained_layers,
                 bce_weight=1.0,
                 gearnet_freeze_layer=0,
                 pretrained_weight_file='ResidueType_lmg_4_512_0.57268.pth',
                 pretrained_weight_has_lm=False,
                 lr=1e-3,
                 batch_size=8,
                 patience=3,
                 use_rus=False,
                 rus_seed=0,
                 undersample_rate=0.1,
                 early_stop_metric='valid_mcc',
                 optimizer='adam',
                 lr_half_epoch=0,
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


    if lr_half_epoch > 0:
        scheduler = ExponentialLR(gamma=0.5**(1/lr_half_epoch), optimizer=pipeline.solver.optimizer)
        pipeline.solver.scheduler = scheduler
    
    train_record, state_dict = pipeline.train_until_fit(
        patience=patience, 
        early_stop_metric=early_stop_metric,
        return_state_dict=True
    )
    return (train_record, state_dict, pipeline)


def read_initial_csv(path):
    try:
        return pd.read_csv(path)
    except (FileNotFoundError, IndexError):
        # File does not exist, or it is empty
        return pd.DataFrame()


def create_single_pred_dataframe(pipeline, dataset):
    df = pd.DataFrame()
    pipeline.task.eval()
    for protein_index, batch in enumerate(data.DataLoader(dataset, batch_size=1, shuffle=False)):
        batch = utils.cuda(batch, device=device())
        label = pipeline.task.target(batch)['label'].flatten()
        
        new_data = {
            'protein_index': protein_index,
            'residue_index': list(range(len(label))),
            'target': label.tolist(),
        }
        pred = pipeline.task.predict(batch).flatten()
        assert(len(label) == len(pred))
        new_data[f'pred'] = [round(t, 5) for t in pred.tolist()]
        new_data = pd.DataFrame(new_data)
        df = pd.concat([df, new_data])
    
    return df

def main(undersample_rate, seed_start, seed_end, lr):
    CSV_PATH = 'rus_pipeline_v2.csv'
    
    for seed in range(seed_start, seed_end):
        bert_freeze_layer = 28
        lr = 4e-4
        print(f'Start {seed} at {pd.Timestamp.now()}')
        patience = 10
        parameters = {
            "bert_freeze_layer": bert_freeze_layer,
            "gearnet_freeze_layer": 0,
            "lr": lr,
            "patience": patience,
            "rus_seed": seed,
            "undersample_rate": undersample_rate,
            "early_stop_metric": "valid_mcc",
            "optimizer": "adam",
            "lr_half_epoch": 12,
        }
        print(parameters)
        result, state_dict, pipeline = run_exp_pure(
            **parameters,
            use_rus=True,
            pretrained_layers=4,
            pretrained_weight_file='../ResidueType_lmg_4_512_0.57268.pth',
            pretrained_weight_has_lm=False,
        )
        max_valid_mcc_row = result[-1-patience]
        new_row = pd.DataFrame.from_dict(
            [{**parameters, **max_valid_mcc_row, 'epoch_count': len(result)}])
        df = read_initial_csv(CSV_PATH)
        df = pd.concat([df, new_row])
        df.to_csv(CSV_PATH, index=False)
        
        # save model
        early_stop_metric_name = parameters['early_stop_metric']

        early_stop_metric = max_valid_mcc_row[early_stop_metric_name]
        csv_prefix = f'rus_{int(undersample_rate*100)}_{seed}_{early_stop_metric:.4f}'
        if early_stop_metric_name == 'valid_bce':
            prefix += '_bce'
        
        pipeline.task.load_state_dict(state_dict)
        pipeline.task.eval()
        
        df_valid = create_single_pred_dataframe(pipeline, pipeline.valid_set)
        df_valid.to_csv(f'preds/{csv_prefix}_valid.csv', index=False)
        
        df_test = create_single_pred_dataframe(pipeline, pipeline.test_set)
        df_test.to_csv(f'preds/{csv_prefix}_test.csv', index=False)


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
