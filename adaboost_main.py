from lib.pipeline import Pipeline
import torch
from torchdrug import utils, data
import pandas as pd
import os

GPU = 1


def read_initial_csv(path):
    try:
        return pd.read_csv(path)
    except (FileNotFoundError, IndexError):
        # File does not exist, or it is empty
        return pd.DataFrame()


def get_train_preds_with_prefixes(prefixes, seed_start=0, seed_end=20):
    if isinstance(prefixes, str):
        prefixes = [prefixes]
    csv_files = [file for file in os.listdir('preds') if file.endswith('.csv')]
    csv_files.sort()
    print(csv_files)
    preds = []
    for seed in range(seed_start, seed_end):
        for prefix in prefixes:
            filtered = [file for file in csv_files if file.startswith(
                f'{prefix}_{seed:02d}_') and 'train' in file]  # get all file with same seed
            if filtered:
                preds.append(f'preds/{filtered[0]}')
    return preds


def aggregate_pred_dataframe(files):
    print(files)
    dfs = [pd.read_csv(f) for f in files]
    final_df = dfs[0].rename(columns={'pred': 'pred_0'})
    for i in range(1, len(dfs)):
        final_df[f'pred_{i}'] = dfs[i]['pred']
    return final_df.reset_index()


def create_mean_ensemble_pred_dataframe(files):
    df = aggregate_pred_dataframe(files)
    df['pred'] = df[[f'pred_{i}' for i in range(len(files))]].mean(axis=1)
    return df


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


def adaboost_iter(iter_num, masks=None, prefix='adaboost', mask_negative_ratio=0.5):
    # initialize new pipeline
    print('Initializing new pipeline')
    model_kwargs = {
        'gpu': GPU,
        'lm_type': 'esm-t33',
        'gearnet_hidden_dim_size': 512,
        'gearnet_hidden_dim_count': 4,
    }
    pipeline = Pipeline(
        model='lm-gearnet',
        dataset='atpbind3d',  # TODO
        gpus=[GPU],
        model_kwargs=model_kwargs,
        optimizer_kwargs={
            'lr': 1e-3,
        },
        undersample_kwargs={
            'masks': masks,
        },
        batch_size=6,  # TODO 6
    )
    pipeline.model.freeze_lm(freeze_all=False, freeze_layer_count=30)

    print('Training..')
    CSV_PATH = f'logs/{prefix}.csv'
    train_record, state_dict = pipeline.train_until_fit(
        patience=5,  # TODO 5
        return_state_dict=True,
        use_dynamic_threshold=False
    )
    df = read_initial_csv(CSV_PATH)
    df = pd.concat([df,
                    pd.DataFrame([{'iter_num': iter_num, **train_record_row}
                                  for train_record_row in train_record])])
    df.to_csv(CSV_PATH, index=False)

    print('Train Done')

    # load the best model
    print('Loading best model')
    pipeline.task.load_state_dict(state_dict)
    pipeline.task.eval()

    # save prediction of current round
    print('Saving prediction')
    df_train = create_single_pred_dataframe(pipeline, pipeline.train_set)
    df_train.to_csv(f'preds/{prefix}_{iter_num:02d}_train.csv', index=False)

    df_valid = create_single_pred_dataframe(pipeline, pipeline.valid_set)
    df_valid.to_csv(f'preds/{prefix}_{iter_num:02d}_valid.csv', index=False)

    df_test = create_single_pred_dataframe(pipeline, pipeline.test_set)
    df_test.to_csv(f'preds/{prefix}_{iter_num:02d}_test.csv', index=False)

    # Get the prediction of all residues with negative labels
    print('Getting prediciton for negative labels')
    if not masks:
        masks = pipeline.dataset.masks

    # fill all the masks with True
    for i in range(len(masks)):
        masks[i].fill_(True)

    all_train_pred_files = get_train_preds_with_prefixes(prefix)
    df_train_pred = create_mean_ensemble_pred_dataframe(all_train_pred_files)
    negative_df = df_train_pred[df_train_pred['target'] == 0]

    confident_negatives = negative_df.sort_values(
        by=['pred'])[:int(len(negative_df) * mask_negative_ratio)]

    print(confident_negatives)

    # for elem in negative_labels[:confident_negatives]:
    #     masks[elem['protein_index']][elem['resudie_index']] = False
    print(confident_negatives.info())
    for index, row in confident_negatives.iterrows():
        index_in_dataset = int(row['protein_index'])
        if row['protein_index'] >= pipeline.dataset.valid_fold()[0]:
            index_in_dataset += len(pipeline.dataset.valid_fold())
        masks[index_in_dataset][int(row['residue_index'])] = False

    return masks


def main(prefix, mask_negative_ratio):
    masks = None
    for i in range(30):
        print(f'Round {i}')
        masks = adaboost_iter(
            i, masks, prefix=prefix, mask_negative_ratio=mask_negative_ratio)
        masks_zero_count = 0
        for mask in masks:
            masks_zero_count += (mask == False).sum().item()
        print(f'Round {i} Done. {masks_zero_count} / 86682 residues masked')


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--prefix', type=str, default='ab_me')
    parser.add_argument('--mask_negative_ratio', type=float, default=0.5)
    args = parser.parse_args()
    GPU = args.gpu
    main(prefix=args.prefix, mask_negative_ratio=args.mask_negative_ratio)
