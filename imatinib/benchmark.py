
import sys
import os

sys.path.insert(0, os.path.abspath('..'))

from lib.pipeline import Pipeline
import numpy as np
import matplotlib.pyplot as plt
import torch
from torchdrug import utils, data
from lib.disable_logger import DisableLogger
from lib.utils import read_initial_csv
import pandas as pd
from lib.pipeline import create_single_pred_dataframe
from lib.utils import aggregate_pred_dataframe, generate_mean_ensemble_metrics_auto


ALL_PARAMS = {
    'esm-t33': {
        'model': 'esm-t33',
        'model_kwargs': {
            'freeze_esm': False,
            'freeze_layer_count': 30,  
        },
        'load_path': None,
    },
    'esm-33-gearnet': {
        'model': 'lm-gearnet',
        'model_kwargs': {
            'lm_type': 'esm-t33',
            'gearnet_hidden_dim_size': 512,
            'gearnet_hidden_dim_count': 4,
            'lm_freeze_layer_count': 30,
        },
        'load_path': None,
    },
    'esm-t33-pretrained': {
        'model': 'esm-t33',
        'model_kwargs': {
            'freeze_esm': False,
            'freeze_layer_count': 30,  
        },
        'load_path': 'esm-t33-pretrained.pth',
    },
    'esm-t33-pretrained-freezelm': {
        'model': 'esm-t33',
        'model_kwargs': {
            'freeze_esm': False,
            'freeze_layer_count': 33,  
        },
        'load_path': 'esm-t33-pretrained.pth',
    },
    'esm-33-gearnet-pretrained': {
        'model': 'lm-gearnet',
        'model_kwargs': {
            'lm_type': 'esm-t33',
            'gearnet_hidden_dim_size': 512,
            'gearnet_hidden_dim_count': 4,
            'lm_freeze_layer_count': 30,
        },
        'load_path': 'lm-gearnet-pretrained.pth',
    },
    'esm-33-gearnet-pretrained-freezelm': {
        'model': 'lm-gearnet',
        'model_kwargs': {
            'lm_type': 'esm-t33',
            'gearnet_hidden_dim_size': 512,
            'gearnet_hidden_dim_count': 4,
            'lm_freeze_layer_count': 33,
        },
        'load_path': 'lm-gearnet-pretrained.pth',
    },
    'esm-33-gearnet-pretrained-freezeall': {
        'model': 'lm-gearnet',
        'model_kwargs': {
            'lm_type': 'esm-t33',
            'gearnet_hidden_dim_size': 512,
            'gearnet_hidden_dim_count': 4,
            'lm_freeze_layer_count': 33,
        },
        'load_path': 'lm-gearnet-pretrained.pth',
        'before_train_lambda': lambda pipeline: pipeline.model.freeze_gearnet(freeze_all=True)
    },
    'esm-33-gearnet-ensemble': {
        'ensemble_count': 10,
        'model_ref': 'esm-33-gearnet',
    },
    'esm-33-gearnet-pretrained-ensemble': {
        'ensemble_count': 10,
        'model_ref': 'esm-33-gearnet-pretrained',
    },
    'esm-33-gearnet-pretrained-freezelm-ensemble': {
        'ensemble_count': 10,
        'model_ref': 'esm-33-gearnet-pretrained-freezelm',
    },
}


def run_test(
    model,
    model_kwargs,
    load_path,
    fold,
    gpu,
    before_train_lambda=None,
    get_inference_df=False,
):
    device = f"cuda:{gpu}"
    pipeline = Pipeline(
        model=model,
        dataset='imatinib',
        gpus=[gpu],
        model_kwargs={
            'gpu': gpu,
            **model_kwargs,
        },
        batch_size=4,
        scheduler='cyclic',
        scheduler_kwargs={
            'base_lr': 3e-4,
            'max_lr': 3e-3,
            'step_size_up': 5,
            'step_size_down': 5,
            'cycle_momentum': False,
        },
        valid_fold_num=fold,
    )
    
    if load_path is not None:
        pipeline.task.load_state_dict(torch.load(load_path, map_location=device))
    
    if before_train_lambda is not None:
        before_train_lambda(pipeline)

    res = pipeline.train_until_fit(max_epoch=10, patience=10)
    if get_inference_df:
        df_valid = create_single_pred_dataframe(pipeline=pipeline, dataset=pipeline.valid_set)
        df_test = create_single_pred_dataframe(pipeline=pipeline, dataset=pipeline.test_set)
        return res[-1], df_valid, df_test
    else:
        return res[-1]

def run_ensemble_test(ensemble_count, model_ref, fold, gpu):
    df_valids = []
    df_tests = []
    for i in range(ensemble_count):
        print(f'ensemble: {i}')
        res, df_valid, df_test = run_test(
            gpu = gpu,
            fold = fold,
            **ALL_PARAMS[model_ref],
            get_inference_df=True,
        )
        df_valids.append(df_valid)
        df_tests.append(df_test)
    
    df_valid = aggregate_pred_dataframe(dfs=df_valids, apply_sig=True)
    df_test = aggregate_pred_dataframe(dfs=df_tests, apply_sig=True)
    
    me_metric = generate_mean_ensemble_metrics_auto(df_valid=df_valid, df_test=df_test, start=0.1, end=0.9, step=0.01)
    del me_metric['best_threshold']
    print(f'me_metric: {me_metric}')
    return me_metric

    
def main(param_key, cnt, gpu):
    for i in range(cnt):
        for fold in range(5):
            print(f'param_key: {param_key}, cnt: {i}, fold: {fold}')
            if ALL_PARAMS[param_key].get('ensemble_count', None) is not None:
                res = run_ensemble_test(
                    ensemble_count = ALL_PARAMS[param_key]['ensemble_count'],
                    model_ref = ALL_PARAMS[param_key]['model_ref'],
                    fold = fold,
                    gpu = gpu,
                )
            else:
                res = run_test(
                    gpu = gpu,
                    fold = fold, 
                    **ALL_PARAMS[param_key],
                )
                
            record_df = read_initial_csv('record.csv')
            record_df = pd.concat([record_df, pd.DataFrame([
                {
                    'model_key': param_key,
                    **res,
                    'fold': fold,
                    'finished_at': pd.Timestamp.now().strftime('%Y-%m-%d %X'),
                }
            ])])
            
            record_df.to_csv('record.csv', index=False)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--param_keys', type=str, nargs='+', default=list(ALL_PARAMS.keys()))
    parser.add_argument('--cnt', type=int, default=1)
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()
    
    for param_key in args.param_keys:
        main(param_key, args.cnt, args.gpu)
    
    
    
    
    
    
