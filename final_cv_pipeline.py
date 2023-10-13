from lib.pipeline import Pipeline
import pandas as pd
from torchdrug import utils, data
import os
import torch
import inspect

from lib.utils import generate_mean_ensemble_metrics_auto, read_initial_csv, aggregate_pred_dataframe
GPU = 0

NEGATIVE_USE_RATIO = 0.25

def rus_preprocess(pipeline):
    # freeze lm
    pipeline.model.freeze_lm(
        freeze_all=False,
        freeze_layer_count=30,
    )
    # build random mask
    masks = pipeline.dataset.masks
    for i in range(len(masks)):
        positive_mask = torch.tensor(pipeline.dataset.targets['binding'][i]) == 1
        negative_mask = torch.rand(len(masks[i])) < NEGATIVE_USE_RATIO
        assert(len(positive_mask) == len(negative_mask))
        mask = positive_mask | negative_mask
        masks[i] = mask
    pipeline.apply_undersample(masks=masks)

def resiboost_preprocess(pipeline, prev_results):
    # freeze lm
    pipeline.model.freeze_lm(
        freeze_all=False,
        freeze_layer_count=30,
    )
    # build mask
    if not prev_results:
        return
    masks = pipeline.dataset.masks
    df_trains = [r['df_train'] for r in prev_results]
    final_df = df_trains[0].rename(columns={'pred': 'pred_0'})
    for i in range(1, len(df_trains)):
        final_df[f'pred_{i}'] = df_trains[i]['pred']
    final_df['pred'] = final_df[[f'pred_{i}' for i in range(len(df_trains))]].mean(axis=1)
    final_df.reset_index(inplace=True)
    negative_df = final_df[final_df['target'] == 0]
    
    confident_negative_df = negative_df.sort_values(
        by=['pred'])[:int(len(negative_df) * (1-NEGATIVE_USE_RATIO))]
    
    for _, row in confident_negative_df.iterrows():
        protein_index_in_dataset = int(row['protein_index'])
        # assume valid fold is consecutive: so that if protein index is larger than first protein index in valid fold, 
        # we need to add the length of valid fold as an offset
        if row['protein_index'] >= pipeline.dataset.valid_fold()[0]:
            protein_index_in_dataset += len(pipeline.dataset.valid_fold())
        masks[protein_index_in_dataset][int(row['residue_index'])] = False
    
    pipeline.apply_undersample(masks=masks)
    
ALL_PARAMS = {
    'esm-t33': {
        'ensemble': False,
        'model': 'esm-t33',
        'model_kwargs': {
            'freeze_esm': False,
            'freeze_layer_count': 30,  
        },
    },
    'bert': {
        'ensemble': False,
        'model': 'bert',
        'model_kwargs': {
            'freeze_bert': False,
            'freeze_layer_count': 29,
        },
    },
    'gearnet': {
        'ensemble': False,
        'model': 'gearnet',
        'model_kwargs': {
            'input_dim': 21,
            'hidden_dims': [512] * 4,
        },
    },
    'bert-gearnet': {
        'ensemble': False,
        'model': 'lm-gearnet',
        'model_kwargs': {
            'lm_type': 'bert',
            'gearnet_hidden_dim_size': 512,
            'gearnet_hidden_dim_count': 4,
        },
        'pipeline_before_train_fn': lambda pipeline: pipeline.model.freeze_lm(
            freeze_all=False,
            freeze_layer_count=29,
        ),
    },
    'esm-33-gearnet': {
        'ensemble': False,
        'model': 'lm-gearnet',
        'model_kwargs': {
            'lm_type': 'esm-t33',
            'gearnet_hidden_dim_size': 512,
            'gearnet_hidden_dim_count': 4,
        },
        'pipeline_before_train_fn': lambda pipeline: pipeline.model.freeze_lm(
            freeze_all=False,
            freeze_layer_count=30,
        ),
    },
    'bert-gearnet-ensemble': {
        'ensemble': True,
        'model': 'lm-gearnet',
        'model_kwargs': {
            'lm_type': 'bert',
            'gearnet_hidden_dim_size': 512,
            'gearnet_hidden_dim_count': 4,
        },
        'pipeline_before_train_fn': lambda pipeline: pipeline.model.freeze_lm(
            freeze_all=False,
            freeze_layer_count=29,
        ),
    },
    'esm-33-gearnet-ensemble': {
        'ensemble': True,
        'model': 'lm-gearnet',
        'model_kwargs': {
            'lm_type': 'esm-t33',
            'gearnet_hidden_dim_size': 512,
            'gearnet_hidden_dim_count': 4,
        },
        'pipeline_before_train_fn': lambda pipeline: pipeline.model.freeze_lm(
            freeze_all=False,
            freeze_layer_count=30,
        ),
    },
    'esm-33-gearnet-ensemble-rus': {
        'ensemble': True,
        'model': 'lm-gearnet',
        'model_kwargs': {
            'lm_type': 'esm-t33',
            'gearnet_hidden_dim_size': 512,
            'gearnet_hidden_dim_count': 4,
        },
        'pipeline_before_train_fn': rus_preprocess,
    },
    'esm-33-gearnet-resiboost': {
        'ensemble': True,
        'model': 'lm-gearnet',
        'model_kwargs': {
            'lm_type': 'esm-t33',
            'gearnet_hidden_dim_size': 512,
            'gearnet_hidden_dim_count': 4,
        },
        'pipeline_before_train_fn': resiboost_preprocess,
        # probably pipeline_before train should receive previous result, so that we can build mask and apply undersample
    },
}

def create_single_pred_dataframe(pipeline, dataset):
    df = pd.DataFrame()
    pipeline.task.eval()
    for protein_index, batch in enumerate(data.DataLoader(dataset, batch_size=1, shuffle=False)):
        batch = utils.cuda(batch, device=f'cuda:{GPU}')
        label = pipeline.task.target(batch)['label'].flatten()

        new_data = {
            'protein_index': protein_index,
            'residue_index': list(range(len(label))),
            'target': label.tolist(),
        }
        pred = pipeline.task.predict(batch).flatten()
        assert (len(label) == len(pred))
        new_data[f'pred'] = [round(t, 5) for t in pred.tolist()]
        new_data = pd.DataFrame(new_data)
        df = pd.concat([df, new_data])

    return df

DEBUG = False
def single_run(
    valid_fold_num,
    model,
    model_kwargs={},
    undersample_kwargs={}, 
    pipeline_before_train_fn=None,
    prev_result=None,
    batch_size=4,
    patience=1 if DEBUG else 5,
):
    pipeline = Pipeline(
        dataset='atpbind3d-minimal' if DEBUG else 'atpbind3d',
        model=model,
        gpus=[GPU],
        model_kwargs={
            'gpu': GPU,
            **model_kwargs,
        },
        undersample_kwargs=undersample_kwargs,
        valid_fold_num=valid_fold_num,
        batch_size=batch_size,
    )
    
    if pipeline_before_train_fn:
        params = inspect.signature(pipeline_before_train_fn).parameters
        if len(params) == 1:
            pipeline_before_train_fn(pipeline)
        else: # used by resiboost, which need previous state
            pipeline_before_train_fn(pipeline, prev_result)
    
    train_record, state_dict = pipeline.train_until_fit(patience=patience, return_state_dict=True)
    
    pipeline.task.load_state_dict(state_dict)
    
    df_train = create_single_pred_dataframe(pipeline, pipeline.train_set)
    df_valid = create_single_pred_dataframe(pipeline, pipeline.valid_set)
    df_test = create_single_pred_dataframe(pipeline, pipeline.test_set)
    
    return {
        'df_train': df_train,
        'df_valid': df_valid,
        'df_test': df_test,
        'record': train_record[-1 - patience],
    }

def write_result(model_key, valid_fold, result):
    # write dataframes to result_cv/{model_key}/fold_{valid_fold}/{train | valid | test}.csv
    # aggregate record to result_cv/result_cv.csv
    folder = f'result_cv/{model_key}/fold_{valid_fold}'
    os.makedirs(folder, exist_ok=True)
    result['df_train'].to_csv(f'{folder}/train.csv', index=False)
    result['df_valid'].to_csv(f'{folder}/valid.csv', index=False)
    result['df_test'].to_csv(f'{folder}/test.csv', index=False)
    record_df = read_initial_csv('result_cv/result_cv.csv')
    record_df = pd.concat([record_df, pd.DataFrame([
        {
            'model_key': model_key,
            'valid_fold': valid_fold,
            **result['record']
        }
    ])])
    record_df.to_csv('result_cv/result_cv.csv', index=False)

def write_result_intermediate(model_key, valid_fold, result, iter):
    folder = f'result_cv/{model_key}/fold_{valid_fold}/intermediate'
    os.makedirs(folder, exist_ok=True)
    result['df_train'].to_csv(f'{folder}/iter_{iter}_train.csv', index=False)
    result['df_valid'].to_csv(f'{folder}/iter_{iter}_valid.csv', index=False)
    result['df_test'].to_csv(f'{folder}/iter_{iter}_test.csv', index=False)
    
    record_df = read_initial_csv(f'{folder}/agg_record.csv')
    record_df = pd.concat([record_df, pd.DataFrame([
        {
            'model_key': model_key,
            'valid_fold': valid_fold,
            'iter': iter,
            **result['record']
        }
    ])])
    record_df.to_csv(f'{folder}/agg_record.csv', index=False)

def write_result_ensemble(model_key, valid_fold, metric):
    record_df = read_initial_csv('result_cv/result_cv.csv')
    record_df = pd.concat([record_df, pd.DataFrame([
        {
            'model_key': model_key,
            'valid_fold': valid_fold,
            **metric
        }
    ])])
    record_df.to_csv('result_cv/result_cv.csv', index=False)

def main(model_key, valid_fold):
    model = ALL_PARAMS[model_key]
    if model['ensemble'] == False:
        result = single_run(
            valid_fold_num=valid_fold,
            **{k: model[k] for k in model if k != 'ensemble'},
        )
        write_result(model_key=model_key,
                     valid_fold=valid_fold,
                     result=result)
    else:
        results = []
        for iter in range(10): # after stable, increase to 100
            result = single_run(
                valid_fold_num=valid_fold,
                **{k: model[k] for k in model if k != 'ensemble'},
                prev_result=results,
            )
            results.append(result)
            write_result_intermediate(
                model_key=model_key,
                valid_fold=valid_fold,
                result=result,
                iter=iter,
            )
        # do mean ensemble
        df_valid = aggregate_pred_dataframe(dfs=[r['df_valid'] for r in results])
        df_test = aggregate_pred_dataframe(dfs=[r['df_test'] for r in results])
        me_metric = generate_mean_ensemble_metrics_auto(df_valid=df_valid, df_test=df_test)
        write_result_ensemble(
            model_key=model_key,
            valid_fold=valid_fold,
            metric=me_metric
        )
        
                


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_keys', type=str, nargs='+', default=['esm-33'])
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--valid_folds', type=int, nargs='+', default=[0, 1, 2, 3, 4])

    args = parser.parse_args()
    GPU = args.gpu
    print(f'Using GPU {GPU}')
    print(f'Running model keys {args.model_keys}')
    print(f'Running valid folds {args.valid_folds}')
    for model_key in args.model_keys:
        for valid_fold in args.valid_folds:
            print(f'Running {model_key} fold {valid_fold}')
            main(model_key=model_key, valid_fold=valid_fold)
