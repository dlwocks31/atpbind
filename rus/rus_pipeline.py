import sys
import os
from filelock import FileLock

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
    'esm_49951': '../esm_t33_gearnet_0.49951.pth',
}

lm_type_map = {
    'bert': 30,
    'esm-t33': 33,
    'esm-t36': 36,
}

def run_exp_pure(bert_freeze_layer,
                 pretrained_layers,
                 pretrained_weight_key,
                 gearnet_freeze_layer,
                 gearnet_hidden_dim_size,
                 lr,
                 batch_size,
                 patience,
                 lr_half_epoch,
                 seed,
                 mlp_lr_ratio,
                 lm_type,
                 bce_weight,
                 gearnet_short_cut,
                 gearnet_concat_hidden,
                 lm_concat_to_output,
                 rus_rate=0.05,
                 rus_by='residue',
                 rus_noise_rate=0,
                 use_rus=True,
                 gpu=None,
                 ):
    gpu = GPU if gpu is None else gpu
    pipeline = Pipeline(
        model='lm-gearnet',
        dataset='atpbind3d',
        gpus=[gpu],
        model_kwargs={
            'gpu': gpu,
            'lm_type' : lm_type,
            'gearnet_hidden_dim_size': gearnet_hidden_dim_size,
            'gearnet_hidden_dim_count': pretrained_layers,
            'gearnet_short_cut': gearnet_short_cut,
            'gearnet_concat_hidden': gearnet_concat_hidden,
            'lm_concat_to_output': lm_concat_to_output,
        },
        optimizer_kwargs={    
            'lr': lr,
        },
        rus_kwargs={
            'rus_seed': seed,
            'rus_rate': rus_rate,
            'rus_noise_rate': rus_noise_rate,
            'rus_by': rus_by,
        } if use_rus else {},
        batch_size=batch_size,
        optimizer='adam',
        bce_weight=bce_weight,
    )
    total_lm_layer = lm_type_map[lm_type]
    pipeline.model.freeze_lm(freeze_all=bert_freeze_layer == total_lm_layer, freeze_layer_count=bert_freeze_layer)
    
    if pretrained_weight_key is not None:
        state_dict = torch.load(pretrained_weight_map[pretrained_weight_key],
                                map_location=f'cuda:{GPU}')
        pipeline.model.gearnet.load_state_dict(state_dict)

    pipeline.model.freeze_gearnet(freeze_layer_count=gearnet_freeze_layer)
    
    # Set mlp lr. Assume mlp param group is added in engine, and thus is last one.
    pipeline.solver.optimizer.param_groups[1]['lr'] = pipeline.solver.optimizer.param_groups[0]['lr'] * mlp_lr_ratio

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
    'lm_type': 'bert',
    'bert_freeze_layer': 28,
    'pretrained_layers': 4,
    'gearnet_freeze_layer': 0,
    'patience': 10,
    'mlp_lr_ratio': 1,
    'lr': 4e-4,
    'lr_half_epoch': 12,
    'pretrained_weight_key': 'rtp_57268',
    'bce_weight': 1,
    'gearnet_hidden_dim_size': 512,
    'gearnet_short_cut': True,
    'gearnet_concat_hidden': True,
    'lm_concat_to_output': False,
}

esm_base_param = {
    'lr': 1e-3,
    'lm_type': 'esm-t33',
    'bert_freeze_layer': 31,
    'pretrained_layers': 4,
    'pretrained_weight_key': None,
    'patience': 5,
}

parameter_by_version = {
    1: {
        'use_rus': False,
    },
    2: {
        'rus_rate': 0.05,
        'rus_by': 'residue',
        'rus_noise_rate': 0.5,
    },
    3: {
        'rus_rate': 0.05,
        'rus_by': 'residue',
        'rus_noise_rate': 0.7,
    },
    4: {
        'rus_rate': 0.05,
        'rus_by': 'residue',
        'rus_noise_rate': 0.3,
    },
    5: {
        'rus_rate': 0.05,
        'rus_by': 'residue',
        'rus_noise_rate': 0.9,
    },
    6: {
        'rus_rate': 0.05,
        'rus_by': 'residue',
        'rus_noise_rate': 1,
    },
    7: {
        'rus_rate': 0.05,
        'rus_by': 'residue',
        'rus_noise_rate': 0.7,
        'lr': 1e-4,
        'mlp_lr_ratio': 5.0,
    },
    # use esm
    8: {
        **esm_base_param,
        'use_rus': False,
    },
    9: {
        **esm_base_param,
        'use_rus': True,
        'rus_rate': 0.05,
        'rus_by': 'residue',
        'rus_noise_rate': 1,
    },
    10: { # v8 + lr_half_epoch 0
        **esm_base_param,
        'lr_half_epoch': 0,
        'use_rus': False,
    },
    11: { # v9 + lr_half_epoch 0
        **esm_base_param,
        'lr_half_epoch': 0,
        'use_rus': True,
        'rus_rate': 0.05,
        'rus_by': 'residue',
        'rus_noise_rate': 1,
    },
    12: { # v11 + patience
        **esm_base_param,
        'lr_half_epoch': 0,
        'use_rus': True,
        'rus_rate': 0.05,
        'rus_by': 'residue',
        'rus_noise_rate': 1,
        'patience': 10,
    },
    13: { # v12 + noise rate 0.7
        **esm_base_param,
        'lr_half_epoch': 0,
        'use_rus': True,
        'rus_rate': 0.05,
        'rus_by': 'residue',
        'rus_noise_rate': 0.7,
        'patience': 10,
    },
    # TRAIN 3 LAYERS
    14: { # No RUS **BASELINE**
        **esm_base_param,
        'lr_half_epoch': 0,
        'use_rus': False,
        'bert_freeze_layer': 30,
    },
    15 : { # RUS Noise Rate 0.7
        **esm_base_param,
        'lr_half_epoch': 0,
        'use_rus': True,
        'rus_rate': 0.05,
        'rus_by': 'residue',
        'rus_noise_rate': 0.7,
        'bert_freeze_layer': 30,
    },
    16 : { # RUS Noise Rate 1
        **esm_base_param,
        'lr_half_epoch': 0,
        'use_rus': True,
        'rus_rate': 0.05,
        'rus_by': 'residue',
        'rus_noise_rate': 1,
        'bert_freeze_layer': 30,
    },
    17 : { # RUS Noise Rate 0.9
        **esm_base_param,
        'lr_half_epoch': 0,
        'use_rus': True,
        'rus_rate': 0.05,
        'rus_by': 'residue',
        'rus_noise_rate': 0.9,
        'bert_freeze_layer': 30,
    },
    18 : { # High RUS Rate with no noise
        **esm_base_param,
        'lr_half_epoch': 0,
        'use_rus': True,
        'rus_rate': 0.5,
        'rus_by': 'residue',
        'rus_noise_rate': 0,
        'bert_freeze_layer': 30,
    },
    19: { # Add pretraining
        **esm_base_param,
        'lr_half_epoch': 0,
        'use_rus': False,
        'bert_freeze_layer': 30,
        'pretrained_weight_key': 'esm_49951',
    },
    20 : { # High RUS Rate, With some noise
        **esm_base_param,
        'lr_half_epoch': 0,
        'use_rus': True,
        'rus_rate': 0.5,
        'rus_by': 'residue',
        'rus_noise_rate': 0.7,
        'bert_freeze_layer': 30,
    },
    21 : { # High RUS Rate, With some noise, and with lr half epoch and patience
        **esm_base_param,
        'lr_half_epoch': 10,
        'patience': 10,
        'use_rus': True,
        'rus_rate': 0.5,
        'rus_by': 'residue',
        'rus_noise_rate': 0.7,
        'bert_freeze_layer': 30,
    },
    22: { # No RUS, bce weight
        **esm_base_param,
        'lr_half_epoch': 0,
        'use_rus': False,
        'bert_freeze_layer': 30,
        'bce_weight': 5,
    },
    23 : { # High RUS Rate, With some noise, and with lr half epoch and patience
        **esm_base_param,
        'lr_half_epoch': 8,
        'patience': 8,
        'use_rus': True,
        'rus_rate': 0.2,
        'rus_by': 'residue',
        'rus_noise_rate': 0.5,
        'bert_freeze_layer': 30,
    },
    # v6 BERT without pretrain
    24: {
        'rus_rate': 0.05,
        'rus_by': 'residue',
        'rus_noise_rate': 1,
        'pretrained_weight_key': None,
    },
    25 : { # High RUS Rate with no noise
        **esm_base_param,
        'lr_half_epoch': 0,
        'use_rus': True,
        'rus_rate': 0.2,
        'rus_by': 'residue',
        'rus_noise_rate': 0,
        'bert_freeze_layer': 30,
    },
    26: { # More GearNet Layer?
        **esm_base_param,
        'lr_half_epoch': 0,
        'use_rus': False,
        'bert_freeze_layer': 30,
        'pretrained_layers': 6,
    },
    27: { # Less GearNet Layer?
        **esm_base_param,
        'lr_half_epoch': 0,
        'use_rus': False,
        'bert_freeze_layer': 30,
        'pretrained_layers': 2,
    },
    28: { # Was short_cut important?
        **esm_base_param,
        'lr_half_epoch': 0,
        'use_rus': False,
        'bert_freeze_layer': 30,
        'gearnet_short_cut': False,
    },
    29: { # Was concat_hidden important?
        **esm_base_param,
        'lr_half_epoch': 0,
        'use_rus': False,
        'bert_freeze_layer': 30,
        'gearnet_concat_hidden': False,
    },
    30: { # Remove both?
        **esm_base_param,
        'lr_half_epoch': 0,
        'use_rus': False,
        'bert_freeze_layer': 30,
        'gearnet_short_cut': False,
        'gearnet_concat_hidden': False,
    },
    31: { # Is concat_to_output important?
        **esm_base_param,
        'lr_half_epoch': 0,
        'use_rus': False,
        'bert_freeze_layer': 30,
        'lm_concat_to_output': True,
    },
}


def main(param_versions, seed_start, seed_end, batch_size):
    CSV_PATH = 'rus_pipeline_v3.csv'
    LOCK_PATH = 'rus_pipeline_v3.lock'
    def process_safe_append_to_csv(new_row):
        lock = FileLock(LOCK_PATH)
        
        with lock:
            df = read_initial_csv(CSV_PATH)
            df = pd.concat([df, new_row])
            df.to_csv(CSV_PATH, index=False)
            
    print(f'Param versions: {param_versions}')
    
    for seed in range(seed_start, seed_end):
        for param_version in param_versions:
            print(f'Start {seed}, v{param_version} at {pd.Timestamp.now()}')
            parameters = {**base_param, **parameter_by_version[param_version], 'seed': seed}
            print({**parameter_by_version[param_version], 'seed': seed})
            result, state_dict, pipeline = run_exp_pure(**parameters, batch_size=batch_size)
            max_valid_mcc_index = max(range(len(result)), key=lambda i: result[i]['valid_mcc'])
            max_valid_mcc_row = result[max_valid_mcc_index]
            new_row = pd.DataFrame.from_dict(
                [{'param_version': param_version, **max_valid_mcc_row, 'epoch_count': len(result)}])
            process_safe_append_to_csv(new_row)
            
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
    parser.add_argument('--param_versions', type=int, nargs='+', default=[1])
    parser.add_argument('--seed_start', type=int, default=0)
    parser.add_argument('--seed_end', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=8)
    args = parser.parse_args()
    GPU = args.gpu
    
    main(param_versions=args.param_versions, seed_start=args.seed_start, seed_end=args.seed_end, batch_size=args.batch_size)
