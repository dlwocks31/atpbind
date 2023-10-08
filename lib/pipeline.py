from copy import deepcopy
from torchdrug import transforms, data, core, layers, tasks, metrics, utils, models
from torchdrug.layers import functional, geometry
from torchdrug.core import Registry as R
import torch
from torch.utils import data as torch_data
from torch.nn import functional as F
import contextlib
import logging
import numpy as np
from functools import cache
import pandas as pd
from datetime import datetime, timedelta

from .tasks import NodePropertyPrediction, MeanEnsembleNodePropertyPrediction
from .datasets import ATPBind, ATPBind3D
from .bert import BertWrapModel, EsmWrapModel
from .custom_models import GearNetWrapModel, LMGearNetModel
from .utils import dict_tensor_to_num, round_dict


class DisableLogger():
    def __enter__(self):
        logging.disable(logging.CRITICAL)

    def __exit__(self, exit_type, exit_value, exit_traceback):
        logging.disable(logging.NOTSET)


def format_timedelta(td: timedelta) -> str:
    total_seconds = int(td.total_seconds())
    minutes, seconds = divmod(total_seconds, 60)

    return f"{minutes}m{seconds}s"


@cache
def get_dataset(dataset, max_length=350):
    print(f'get dataset {dataset}')
    if dataset == 'atpbind3d' or dataset == 'atpbind3d-minimal':
        truncuate_transform = transforms.TruncateProtein(
            max_length=max_length, random=False)
        protein_view_transform = transforms.ProteinView(view='residue')
        transform = transforms.Compose(
            [truncuate_transform, protein_view_transform])

        limit = -1 if dataset == 'atpbind3d' else 5
        return ATPBind3D(transform=transform, limit=limit)
    elif dataset == 'atpbind':
        raise NotImplementedError('atpbind dataset dropped')


def create_single_pred_dataframe(pipeline, dataset):
    df = pd.DataFrame()
    pipeline.task.eval()
    for protein_index, batch in enumerate(data.DataLoader(dataset, batch_size=1, shuffle=False)):
        batch = utils.cuda(batch, device=torch.device(
            f'cuda:{pipeline.gpus[0]}'))
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


METRICS_USING = ("sensitivity", "specificity", "accuracy",
                 "precision", "mcc", "micro_auroc",)


class Pipeline:
    possible_models = ['bert', 'gearnet', 'lm-gearnet',
                       'cnn', 'esm-t6', 'esm-t12', 'esm-t30', 'esm-t33', 'esm-t36', 'esm-t48']
    possible_datasets = ['atpbind', 'atpbind3d', 'atpbind3d-minimal']
    threshold = 0

    def __init__(self,
                 model,
                 dataset,
                 gpus,
                 task='npp',
                 model_kwargs={},
                 optimizer_kwargs={},
                 task_kwargs={},
                 undersample_kwargs={},
                 batch_size=1,
                 bce_weight=1,
                 verbose=False,
                 optimizer='adam',
                 valid_fold_num=0,
                 max_length=350,
                 ):
        self.gpus = gpus

        if model not in self.possible_models and not isinstance(model, torch.nn.Module):
            raise ValueError(
                'Model must be one of {}, or is torch.nn.Module'.format(self.possible_models))

        if dataset not in self.possible_datasets:
            raise ValueError('Dataset must be one of {}'.format(
                self.possible_datasets))

        self.load_model(model, **model_kwargs)

        self.dataset = get_dataset(dataset, max_length=max_length)
        self.valid_fold_num = valid_fold_num
        self.train_set, self.valid_set, self.test_set = self.dataset.initialize_undersampling(
            **undersample_kwargs).split(valid_fold_num=valid_fold_num)
        print("train samples: %d, valid samples: %d, test samples: %d" %
              (len(self.train_set), len(self.valid_set), len(self.test_set)))

        if dataset == 'atpbind':
            self.task = NodePropertyPrediction(
                self.model,
                normalization=False,
                num_mlp_layer=2,
                metric=METRICS_USING,
                bce_weight=torch.tensor(
                    [bce_weight], device=torch.device(f'cuda:{self.gpus[0]}')),
                **task_kwargs,
            )
        elif dataset == 'atpbind3d' or dataset == 'atpbind3d-minimal':
            edge_layers = [
                geometry.SpatialEdge(radius=10.0, min_distance=5),
                geometry.KNNEdge(k=10, min_distance=5),
                geometry.SequentialEdge(max_distance=2),
            ]

            graph_construction_model = layers.GraphConstruction(
                node_layers=[geometry.AlphaCarbonNode()],
                edge_layers=edge_layers,
                edge_feature="gearnet"
            )
            task_kwargs = {
                'graph_construction_model': graph_construction_model,
                'normalization': False,
                'num_mlp_layer': 2,
                'metric': METRICS_USING,
                'bce_weight': torch.tensor([bce_weight], device=torch.device(f'cuda:{self.gpus[0]}')),
                **task_kwargs,
            }

            if task == 'npp':
                self.task = NodePropertyPrediction(
                    self.model,
                    **task_kwargs,
                )
            elif task == 'mean-ensemble':
                self.task = MeanEnsembleNodePropertyPrediction(
                    self.model,
                    **task_kwargs,
                )
            else:
                raise ValueError(
                    'Task must be one of {}'.format(['npp', 'me_npp']))

        if not optimizer in ['adam', 'adamw']:
            raise ValueError(
                'Optimizer must be one of {}'.format(['adam', 'adamw']))
        # it does't matter whether we use self.task or self.model.parameters(), since mlp is added at preprocess time
        # and mlp parameters is then added to optimizer
        if optimizer == 'adam':
            self.optimizer = torch.optim.Adam(
                self.model.parameters(), **optimizer_kwargs)
        elif optimizer == 'adamw':
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(), **optimizer_kwargs)

        self.verbose = verbose
        self.batch_size = batch_size
        with DisableLogger():
            self.solver = core.Engine(self.task,
                                      self.train_set,
                                      self.valid_set,
                                      self.test_set,
                                      self.optimizer,
                                      batch_size=self.batch_size,
                                      log_interval=1000000000,
                                      gpus=self.gpus,
                                      )

    def apply_undersample(self, masks):
        self.train_set, self.valid_set, self.test_set = self.dataset.initialize_undersampling(
            masks=masks).split(valid_fold_num=self.valid_fold_num)
        print("train samples: %d, valid samples: %d, test samples: %d" %
              (len(self.train_set), len(self.valid_set), len(self.test_set)))
        with DisableLogger():
            self.solver = core.Engine(self.task,
                                      self.train_set,
                                      self.valid_set,
                                      self.test_set,
                                      self.optimizer,
                                      batch_size=self.batch_size,
                                      log_interval=1000000000,
                                      gpus=self.gpus,
                                      )

    def load_model(self, model, **model_kwargs):
        print(f'load model {model}, kwargs: {model_kwargs}')
        with DisableLogger():
            if model == 'bert':
                self.model = BertWrapModel(**model_kwargs)
            elif model == 'gearnet':
                self.model = GearNetWrapModel(**model_kwargs)
            elif model == 'lm-gearnet':
                self.model = LMGearNetModel(**model_kwargs)
            elif model == 'cnn':
                self.model = models.ProteinCNN(**model_kwargs)
            elif model.startswith('esm'):
                self.model = EsmWrapModel(model_type=model, **model_kwargs)
            # pre built model, eg LoraModel. I wonder wheter there is better way to check this
            elif isinstance(model, torch.nn.Module):
                self.model = model

    def train(self, num_epoch):
        return self.solver.train(num_epoch=num_epoch)

    def train_until_fit(self, patience=1, early_stop_metric='valid_mcc', return_preds=False, return_state_dict=False, use_dynamic_threshold=True):
        if early_stop_metric not in ['valid_mcc', 'valid_bce']:
            raise ValueError('early_stop_metric must be one of {}'.format(
                ['valid_mcc', 'valid_bce']))
        from itertools import count
        train_record = []
        train_preds = None
        valid_preds = None
        test_preds = None
        state_dict = None
        best_metric = -1 if early_stop_metric == 'valid_mcc' else 1e10

        last_time = datetime.now()
        for epoch in count(start=1):
            cm = contextlib.nullcontext() if self.verbose else DisableLogger()
            with cm:
                self.train(num_epoch=1)

                # record
                cur_result = self.evaluate(split='test', threshold=0)
                cur_result['train_bce'] = self.get_last_bce()
                cur_result['valid_bce'] = self.calculate_valid_loss()
                if use_dynamic_threshold:
                    cur_result['valid_mcc'] = self.calculate_best_mcc_and_threshold(
                        threshold_set='valid'
                    )['best_mcc']
                else:
                    cur_result['valid_mcc'] = self.evaluate(
                        split='valid', threshold=0)['mcc']
                cur_result = round_dict(cur_result, 4)
                train_record.append(cur_result)
                # logging
                cur_time = datetime.now()
                print(f'{format_timedelta(cur_time - last_time)} {cur_result}')
                last_time = cur_time
                # early stop
                should_replace_best_metric = cur_result[
                    'valid_mcc'] > best_metric if early_stop_metric == 'valid_mcc' else cur_result['valid_bce'] < best_metric
                if should_replace_best_metric:
                    if return_preds:
                        best_metric = cur_result[early_stop_metric]
                        train_preds = create_single_pred_dataframe(
                            self, self.train_set)
                        valid_preds = create_single_pred_dataframe(
                            self, self.valid_set)
                        test_preds = create_single_pred_dataframe(
                            self, self.test_set)
                    elif return_state_dict:
                        state_dict = deepcopy(self.task.state_dict())
                        
                best_index = np.argmax([record[early_stop_metric] for record in train_record]) if early_stop_metric == 'valid_mcc' else np.argmin(
                    [record[early_stop_metric] for record in train_record])
                if best_index < len(train_record) - patience:
                    break


        if return_preds:
            return (train_record, train_preds, valid_preds, test_preds)
        elif return_state_dict:
            return (train_record, state_dict)
        else:
            return train_record

    def get_last_bce(self):
        from statistics import mean
        meter = self.solver.meter
        index = slice(meter.epoch2batch[-2], meter.epoch2batch[-1])
        bce_records = meter.records['binary cross entropy'][index]
        return mean(bce_records)

    def calculate_valid_loss(self):
        from statistics import mean
        dataloader = data.DataLoader(
            self.valid_set, batch_size=self.batch_size, shuffle=False)
        model = self.task

        model.eval()

        metrics = []
        with torch.no_grad():
            for batch in dataloader:
                batch = utils.cuda(batch, device=f'cuda:{self.gpus[0]}')
                loss, metric = model(batch)
                metrics.append(metric['binary cross entropy'].item())

        return mean(metrics)

    def calculate_best_mcc_and_threshold(self, threshold_set='valid'):
        dataloader = data.DataLoader(
            self.valid_set if threshold_set == 'valid' else self.test_set,
            batch_size=self.batch_size,
            shuffle=False
        )

        preds = []
        targets = []
        thresholds = np.linspace(-3, 1, num=41)
        mcc_values = [0 for i in range(len(thresholds))]
        self.model.eval()
        with torch.no_grad():
            for batch in dataloader:
                batch = utils.cuda(
                    batch, device=torch.device(f'cuda:{self.gpus[0]}'))
                pred, target = self.task.predict_and_target(batch)
                preds.append(pred)
                targets.append(target)

        pred = utils.cat(preds)
        target = utils.cat(targets)

        for i, threshold in enumerate(thresholds):
            mcc = self.task.evaluate(
                pred, target, threshold
            )['mcc']
            mcc_values[i] = mcc_values[i] + mcc

        max_mcc_idx = np.argmax(mcc_values)

        return {
            'best_mcc': mcc_values[max_mcc_idx],
            'best_threshold': thresholds[max_mcc_idx]
        }

    def evaluate(self, split="test", verbose=False, threshold=0):
        if threshold == 'auto':
            mcc_and_threshold = self.calculate_best_mcc_and_threshold(
                threshold_set='valid')
            if verbose:
                print(f'threshold: {mcc_and_threshold}\n')
            self.task.threshold = mcc_and_threshold['best_threshold']
        else:
            self.task.threshold = threshold
        return dict_tensor_to_num(self.solver.evaluate(split=split))
