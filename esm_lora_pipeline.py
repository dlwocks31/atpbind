from transformers import AutoTokenizer, EsmModel
from torchdrug import core
from peft import LoraConfig, TaskType, LoraModel
import torch
from lib.pipeline import Pipeline
import pandas as pd
from itertools import product


class EsmRawWrapModel(torch.nn.Module, core.Configurable):
    def __init__(self, 
                 gpu,
                 esm_tokenizer, esm_model):
        super().__init__()
        self.esm_tokenizer = esm_tokenizer
        self.esm_model = esm_model
        self.input_dim = 21
        self.output_dim = self.esm_model.config.hidden_size
        self.gpu = gpu

    def forward(self, graph, _, all_loss=None, metric=None):
        input = [seq.replace('.', ' ') for seq in graph.to_sequence()]
        input_len = [len(seq.replace(' ', '')) for seq in input]

        encoded_input = self.esm_tokenizer(
            input, return_tensors='pt', padding=True).to(f'cuda:{gpu}')
        embedding_rpr = self.esm_model(**encoded_input)
        
        residue_feature = []
        for i, emb in enumerate(embedding_rpr.last_hidden_state):
            # skip residue feature for [CLS] and [SEP], since they are not in the original sequence
            residue_feature.append(emb[1:1+input_len[i]])
        
        x = torch.cat(residue_feature)

        return {"residue_feature": x}
    
    
    
def freeze_lora_esm(lora_model, train_layer):
    for param in lora_model.encoder.layer[:-train_layer].parameters():
        param.requires_grad = False

def run_exp_pure(
    lm_key,
    r,
    lora_dropout,
    early_stop_metric,
    train_layer,
    bias,
    patience,
    lr,
    gpu,
):
    tokenizer = AutoTokenizer.from_pretrained(lm_key)
    esm_model = EsmModel.from_pretrained(lm_key).to(f"cuda:{gpu}")
    peft_config = LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION, 
        inference_mode=False, 
        r=r, 
        lora_alpha=r * 4,
        target_modules=["query", "key", "value"],
        lora_dropout=lora_dropout,
        bias=bias,
    )
    lora_model = LoraModel(esm_model, peft_config, "default").to(f"cuda:{gpu}")
    
    freeze_lora_esm(lora_model, train_layer)
    esm_raw_wrap = EsmRawWrapModel(gpu, tokenizer, lora_model)
    
    
    pipeline = Pipeline(
        esm_raw_wrap,
        'atpbind3d',
        batch_size=4,
        gpus=[gpu],
        optimizer_kwargs={
            'lr': lr,
        }
    )
    
    train_record = pipeline.train_until_fit(
        patience=patience, 
        early_stop_metric=early_stop_metric
    )
    
    return train_record
    
    
def read_initial_csv(path):
    try:
        return pd.read_csv(path)
    except (FileNotFoundError, IndexError):
        # File does not exist, or it is empty
        return pd.DataFrame()

def run_and_save(params, max_exp):
    print(pd.Timestamp.now(), params)

    CSV_PATH = 'esm_lora_pipeline.csv'
    df = read_initial_csv(CSV_PATH)
    # Filter columns that exist in params and check if a matching record is present
    mask = (df[list(params.keys())] == pd.Series(params)).all(axis=1)
    if mask.sum() >= max_exp:
        print("Too many row with same param, Skipping run_exp_pure.")
        return  # Exit the function

    result = run_exp_pure(**params)
    
    df = read_initial_csv(CSV_PATH)
    new_row = pd.DataFrame.from_dict([{
        **params, 
        **result[-1-params['patience']],
        'epoch_count': len(result),
    }])
    df = pd.concat([df, new_row])
    df.to_csv(CSV_PATH, index=False)


def main(gpu, max_exp):
    r_list = [4, 6, 8, 10, 12, 14, 16]
    lora_dropout_list = [0, 0.1]
    early_stop_metric = 'valid_mcc'
    train_layer_list = [33, 16]
    bias_list = ['none', 'all']
    lm_key = "facebook/esm2_t33_650M_UR50D"
    lr_list = [1e-3, 5e-4]
    patience = 3
    for r, lora_dropout, train_layer, bias, lr in product(
        r_list,
        lora_dropout_list,
        train_layer_list,
        bias_list,
        lr_list,
    ):
        params = {
            "lm_key": lm_key,
            "r": r,
            "lora_dropout": lora_dropout,
            "early_stop_metric": early_stop_metric,
            "train_layer": train_layer,
            "bias": bias,
            "patience": patience,
            "lr": lr,
            "gpu": gpu,
        }
        run_and_save(params, max_exp)
    

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int)
    parser.add_argument('--max_exp', type=int)
    args = parser.parse_args()
    gpu = args.gpu
    max_exp = args.max_exp
    main(gpu=gpu, max_exp=max_exp)