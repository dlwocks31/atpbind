from lib.pipeline import Pipeline
import torch
from torchdrug import utils, data
import pandas as pd

GPU = 1

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
        batch = utils.cuda(batch, device=f'cuda:{GPU}')
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

def adaboost_iter(iter_num, masks=None, prefix='adaboost'):
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
        dataset='atpbind3d', # TODO
        gpus=[GPU],
        model_kwargs=model_kwargs,
        optimizer_kwargs={
            'lr': 1e-3,
        },
        undersample_kwargs={
            'masks': masks,
        },
        batch_size=6, # TODO 6
    )
    pipeline.model.freeze_lm(freeze_all=False, freeze_layer_count=30)
    
    print('Training..')
    CSV_PATH = f'logs/adaboost.csv'
    train_record, state_dict = pipeline.train_until_fit(
        patience=5, # TODO 5 
        return_state_dict=True,
        use_dynamic_threshold=False
    )
    df = read_initial_csv(CSV_PATH)
    df = pd.concat([df, 
                    pd.DataFrame([{'iter_num': iter_num, **train_record_row} 
                                  for train_record_row in train_record])])
    df.to_csv(CSV_PATH, index=False)

    print('Train Done')
    train_dataloader = data.DataLoader(pipeline.train_set, batch_size=1, shuffle=False)

    # load the best model
    print('Loading best model')
    pipeline.task.load_state_dict(state_dict)
    pipeline.task.eval()


    # Get the prediction of all residues with negative labels
    print('Getting prediciton for negative labels')
    if not masks:
        masks = pipeline.dataset.masks

    negative_labels = []
    for protein_index, batch in enumerate(train_dataloader):
        index_in_dataset = protein_index if protein_index < pipeline.dataset.valid_fold()[0] else protein_index + len(pipeline.dataset.valid_fold())
        batch = utils.cuda(batch, device=f'cuda:{GPU}')
        label = pipeline.task.target(batch)['label'].flatten()
        pred = pipeline.task.predict(batch).flatten()
        for i in range(len(label)):
            if label[i] == 0 and masks[index_in_dataset][i]:
                negative_labels.append({
                    "protein_index": index_in_dataset,
                    'resudie_index': i,
                    'pred': pred[i].item(),
                })
            
    negative_labels = sorted(negative_labels, key=lambda x: x['pred'], reverse=False)
    top_10_percent = int(len(negative_labels) * 0.1) # TODO int(len(negative_labels) * 0.1)
    for elem in negative_labels[:top_10_percent]:
        masks[elem['protein_index']][elem['resudie_index']] = False

    # save prediction of current round
    print('Saving prediction')
    df_valid = create_single_pred_dataframe(pipeline, pipeline.valid_set)
    df_valid.to_csv(f'preds/{prefix}_{iter_num:02d}_valid.csv', index=False)

    df_test = create_single_pred_dataframe(pipeline, pipeline.test_set)
    df_test.to_csv(f'preds/{prefix}_{iter_num:02d}_test.csv', index=False)
    
    return masks
    

def main(prefix):
    masks = None
    for i in range(30):
        print(f'Round {i}')
        masks = adaboost_iter(i, masks, prefix)
        masks_zero_count = 0
        for mask in masks:
            masks_zero_count += (mask == False).sum().item()
        print(f'Round {i} Done. {masks_zero_count} / 86682 residues masked')

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=1)
    parser.add_argument('--prefix', type=str, default='adaboost')
    args = parser.parse_args()
    GPU = args.gpu
    main(prefix=args.prefix)