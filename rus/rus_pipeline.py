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

pretrained_weight_map = {
    'rtp_57268': '../ResidueType_lmg_4_512_0.57268.pth',
}

def run_exp_pure(bert_freeze_layer,
                 pretrained_layers,
                 pretrained_weight_key,
                 gearnet_freeze_layer,
                 lr,
                 batch_size,
                 patience,
                 lr_half_epoch,
                 seed,
                 rus_rate=0.05,
                 rus_by='residue',
                 use_rus=True,
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
        rus_kwargs={
            'rus_seed': seed,
            'rus_rate': rus_rate,
            'rus_by': rus_by,
        } if use_rus else {},
        batch_size=batch_size,
        optimizer='adam',
    )
    if pretrained_weight_key is not None:
        state_dict = torch.load(pretrained_weight_map[pretrained_weight_key],
                                map_location=f'cuda:{GPU}')
        pipeline.model.gearnet.load_state_dict(state_dict)

    pipeline.model.freeze_gearnet(freeze_layer_count=gearnet_freeze_layer)


    if lr_half_epoch > 0:
        scheduler = ExponentialLR(gamma=0.5**(1/lr_half_epoch), optimizer=pipeline.solver.optimizer)
        pipeline.solver.scheduler = scheduler
    
    train_record, state_dict = pipeline.train_until_fit(
        patience=patience, 
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


base_param = {
    'bert_freeze_layer': 28,
    'pretrained_layers': 4,
    'gearnet_freeze_layer': 0,
    'patience': 10,
    'batch_size': 8,
}

parameter_by_version = {
    1: {
        'lr': 4e-4,
        'lr_half_epoch': 12,
        'use_rus': False,
        'pretrained_weight_key': 'rtp_57268',
    }
}
def main(param_version, seed_start, seed_end):
    CSV_PATH = 'rus_pipeline_v3.csv'
    
    for seed in range(seed_start, seed_end):
        print(f'Start {seed} at {pd.Timestamp.now()}')
        patience = 10
        parameters = {**base_param, **parameter_by_version[param_version], 'seed': seed}
        print({**parameter_by_version[param_version], 'seed': seed})
        result, state_dict, pipeline = run_exp_pure(**parameters)
        max_valid_mcc_row = result[-1-patience]
        new_row = pd.DataFrame.from_dict(
            [{'param_version': param_version, **max_valid_mcc_row, 'epoch_count': len(result)}])
        df = read_initial_csv(CSV_PATH)
        df = pd.concat([df, new_row])
        df.to_csv(CSV_PATH, index=False)
        
        # save model
        early_stop_metric_name = 'valid_mcc'

        early_stop_metric = max_valid_mcc_row[early_stop_metric_name]
        csv_prefix = f'v{param_version:03d}_{seed}_{early_stop_metric:.4f}'
        
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
    parser.add_argument('--param_version', type=int, default=1)
    parser.add_argument('--seed_start', type=int, default=0)
    parser.add_argument('--seed_end', type=int, default=10)
    args = parser.parse_args()
    GPU = args.gpu
    main(param_version=args.param_version, seed_start=args.seed_start, seed_end=args.seed_end)
