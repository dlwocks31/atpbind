from lib.pipeline import Pipeline, create_single_pred_dataframe
from torchdrug import utils, data
import pandas as pd
import os
import re

GPU = 1


def read_initial_csv(path):
    try:
        return pd.read_csv(path)
    except (FileNotFoundError, IndexError):
        # File does not exist, or it is empty
        return pd.DataFrame()


def get_train_preds_with_prefix(prefix):
    csv_files = [file for file in os.listdir('preds') if file.endswith('.csv')]
    csv_files.sort()
    filtered = [file for file in csv_files if re.match(
        f'{prefix}_\d+_train\.csv', file)]
    return [f'preds/{i}' for i in filtered]


def aggregate_pred_dataframe(files):
    dfs = [pd.read_csv(f) for f in files]
    final_df = dfs[0].rename(columns={'pred': 'pred_0'})
    for i in range(1, len(dfs)):
        final_df[f'pred_{i}'] = dfs[i]['pred']
    return final_df.reset_index()


def create_mean_ensemble_pred_dataframe(files):
    df = aggregate_pred_dataframe(files)
    df['pred'] = df[[f'pred_{i}' for i in range(len(files))]].mean(axis=1)
    return df


def build_mask_from_prefix_me(masks, prefix, mask_negative_ratio, valid_fold):
    # fill all the masks with True
    for i in range(len(masks)):
        masks[i].fill_(True)

    all_train_pred_files = get_train_preds_with_prefix(prefix)
    if not all_train_pred_files:
        print('No files found')
        return masks
    print('Building mask from files: ', all_train_pred_files)
    df_train_pred = create_mean_ensemble_pred_dataframe(all_train_pred_files)
    negative_df = df_train_pred[df_train_pred['target'] == 0]

    confident_negatives = negative_df.sort_values(
        by=['pred'])[:int(len(negative_df) * mask_negative_ratio)]

    for _, row in confident_negatives.iterrows():
        index_in_dataset = int(row['protein_index'])
        # assume valid fold is consecutive
        if row['protein_index'] >= valid_fold[0]:
            index_in_dataset += len(valid_fold)
        masks[index_in_dataset][int(row['residue_index'])] = False

    return masks

CYCLE_SIZE = 10
CYCLIC_SCHEDULER_KWARGS = {
    'scheduler': 'cyclic',
    'scheduler_kwargs': {
        'base_lr': 3e-4,
        'max_lr': 3e-3,
        'step_size_up': CYCLE_SIZE / 2,
        'step_size_down': CYCLE_SIZE / 2,
        'cycle_momentum': False
    }
}


def adaboost_iter(iter_num, masks=None, prefix='adaboost', mask_negative_ratio=0.5):
    # initialize new pipeline
    print('Initializing new pipeline')
    model_kwargs = {
        'gpu': GPU,
        'lm_type': 'esm-t33',
        'gearnet_hidden_dim_size': 512,
        'gearnet_hidden_dim_count': 4,
        'lm_freeze_layer_count': 30,
    }
    pipeline = Pipeline(
        model='lm-gearnet',
        dataset='atpbind3d',  # TODO
        gpus=[GPU],
        model_kwargs=model_kwargs,
        batch_size=6,  # TODO 6
        **CYCLIC_SCHEDULER_KWARGS,
    )

    print('Apply undersample')
    masks = build_mask_from_prefix_me(
        masks=pipeline.dataset.masks,
        prefix=prefix,
        mask_negative_ratio=mask_negative_ratio,
        valid_fold=pipeline.dataset.valid_fold()
    )
    pipeline.apply_mask_and_weights(masks=masks)

    print('Training..')
    pipeline.train_until_fit(patience=CYCLE_SIZE, max_epoch=CYCLE_SIZE)
    print('Train Done')
    train_preds = create_single_pred_dataframe(pipeline=pipeline, dataset=pipeline.train_set)
    valid_preds = create_single_pred_dataframe(pipeline=pipeline, dataset=pipeline.valid_set)
    test_preds = create_single_pred_dataframe(pipeline=pipeline, dataset=pipeline.test_set)
    # save prediction of current round
    print('Saving prediction')
    train_preds.to_csv(f'preds/{prefix}_{iter_num:02d}_train.csv', index=False)
    valid_preds.to_csv(f'preds/{prefix}_{iter_num:02d}_valid.csv', index=False)
    test_preds.to_csv(f'preds/{prefix}_{iter_num:02d}_test.csv', index=False)


def main(prefix, mask_negative_ratio, max_iter):
    train_preds = get_train_preds_with_prefix(prefix)
    print(f'train_preds: {train_preds}')
    cur_len = len(get_train_preds_with_prefix(prefix))
    for i in range(cur_len, max_iter):
        print(f'Round {i}')
        adaboost_iter(i, prefix=prefix,
                      mask_negative_ratio=mask_negative_ratio)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--prefix', type=str, default='ab_me')
    parser.add_argument('--mask_negative_ratio', type=float, default=0.5)
    parser.add_argument('--max_iter', type=int, default=50)
    args = parser.parse_args()
    GPU = args.gpu
    prefix = f'{args.prefix}_{int(args.mask_negative_ratio*100)}'
    print(f'Use prefix {prefix}')
    main(prefix=prefix,
         mask_negative_ratio=args.mask_negative_ratio,
         max_iter=args.max_iter
         )
