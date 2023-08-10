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

from .tasks import NodePropertyPrediction, MeanEnsembleNodePropertyPrediction
from .datasets import ATPBind, ATPBind3D
from .bert import BertWrapModel
from .custom_models import GearNetWrapModel, LMGearNetModel
from .utils import dict_tensor_to_num, round_dict

class DisableLogger():
    def __enter__(self):
       logging.disable(logging.CRITICAL)
    def __exit__(self, exit_type, exit_value, exit_traceback):
       logging.disable(logging.NOTSET)

@cache
def get_dataset(dataset):
    print(f'get dataset {dataset}')
    if dataset == 'atpbind':
        truncuate_transform = transforms.TruncateProtein(max_length=350, random=False)
        protein_view_transform = transforms.ProteinView(view='residue')
        transform = transforms.Compose([truncuate_transform, protein_view_transform])

        dataset = ATPBind(atom_feature=None, bond_feature=None,
                        residue_feature="default", transform=transform)

        train_set, valid_set, test_set = dataset.split()
        print("train samples: %d, valid samples: %d, test samples: %d" %
            (len(train_set), len(valid_set), len(test_set)))
        
        return train_set, valid_set, test_set
    elif dataset == 'atpbind3d' or dataset == 'atpbind3d-minimal':
        truncuate_transform = transforms.TruncateProtein(max_length=350, random=False)
        protein_view_transform = transforms.ProteinView(view='residue')
        transform = transforms.Compose([truncuate_transform, protein_view_transform])

        if dataset == 'atpbind3d':
            dataset = ATPBind3D(transform=transform)
        elif dataset == 'atpbind3d-minimal':
            dataset = ATPBind3D(transform=transform, limit=5)

        train_set, valid_set, test_set = dataset.split()
        print("train samples: %d, valid samples: %d, test samples: %d" %
              (len(train_set), len(valid_set), len(test_set)))
        
        return train_set, valid_set, test_set

METRICS_USING = ("sensitivity", "specificity", "accuracy", "precision", "mcc", "micro_auroc",)
class Pipeline:
    possible_models = ['bert', 'gearnet', 'lm-gearnet', 'cnn']
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
                 graph_knn_k=10,
                 graph_spatial_radius=10.0,
                 graph_sequential_max_distance=2,
                 batch_size=1,
                 bce_weight=1,
                 verbose=False,
                 ):
        self.gpus = gpus

        if model not in self.possible_models:
            raise ValueError('Model must be one of {}'.format(self.possible_models))
    
        if dataset not in self.possible_datasets:
            raise ValueError('Dataset must be one of {}'.format(self.possible_datasets))
           
        with DisableLogger():     
            if model == 'bert':
                self.model = BertWrapModel(**model_kwargs)
            elif model == 'gearnet':
                self.model = GearNetWrapModel(graph_sequential_max_distance=graph_sequential_max_distance, **model_kwargs)
            elif model == 'lm-gearnet':
                self.model = LMGearNetModel(graph_sequential_max_distance=graph_sequential_max_distance, **model_kwargs)
            elif model == 'cnn':
                self.model = models.ProteinCNN(**model_kwargs)
        
        self.train_set, self.valid_set, self.test_set = get_dataset(dataset)
        
        if dataset == 'atpbind':
            self.task = NodePropertyPrediction(
                self.model, 
                normalization=False,
                num_mlp_layer=2,
                metric=METRICS_USING,
                bce_weight=torch.tensor([bce_weight], device=torch.device(f'cuda:{self.gpus[0]}')),
                **task_kwargs,
            )
        elif dataset == 'atpbind3d' or dataset == 'atpbind3d-minimal':
            edge_layers = [
                geometry.SpatialEdge(radius=graph_spatial_radius, min_distance=5),
                geometry.KNNEdge(k=graph_knn_k, min_distance=5),
                geometry.SequentialEdge(max_distance=graph_sequential_max_distance),
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
                raise ValueError('Task must be one of {}'.format(['npp', 'me_npp']))



        
        optimizer = torch.optim.Adam(self.model.parameters(), **optimizer_kwargs)
        with DisableLogger():
            self.solver = core.Engine(self.task,
                                        self.train_set,
                                        self.valid_set,
                                        self.test_set,
                                        optimizer,
                                        batch_size=batch_size,
                                        log_interval=1000000000,
                                        gpus=gpus
            )
        
        self.verbose = verbose
        self.batch_size = batch_size
        
    def train(self, num_epoch):
        return self.solver.train(num_epoch=num_epoch)
    
    def train_until_fit(self, patience=1, return_state_dict=False):
        from itertools import count
        train_record = []
        best_state_dict = None
        best_valid_mcc = -1
        for epoch in count(start=1):
            cm = contextlib.nullcontext() if self.verbose else DisableLogger()
            with cm:
                self.train(num_epoch=1)
                cur_result = self.evaluate()
                cur_result['train_bce'] = self.get_last_bce()
                cur_result['valid_bce'] = self.calculate_valid_loss()
                cur_result['valid_mcc'] = self.calculate_best_mcc_and_threshold(
                    threshold_set='valid'
                )['best_mcc']
                cur_result = round_dict(cur_result, 4)
                train_record.append(cur_result)
                print(cur_result)
                if return_state_dict and cur_result['valid_mcc'] > best_valid_mcc:
                    best_valid_mcc = cur_result['valid_mcc']
                    best_state_dict = deepcopy(self.task.state_dict())
                max_mcc_index = np.argmax([record['valid_mcc'] for record in train_record])
                if max_mcc_index < len(train_record) - patience:
                    break
        if return_state_dict:
            return (train_record, best_state_dict)
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
        dataloader = data.DataLoader(self.valid_set, batch_size=self.batch_size, shuffle=False)
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
                batch = utils.cuda(batch, device=torch.device(f'cuda:{self.gpus[0]}'))
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


    def evaluate(self, threshold_set='valid', verbose=False):
        mcc_and_threshold = self.calculate_best_mcc_and_threshold(threshold_set)
        if verbose:
            print(f'threshold: {mcc_and_threshold}\n')
        self.task.threshold = mcc_and_threshold['best_threshold']
        return dict_tensor_to_num(self.solver.evaluate("test"))
