from torchdrug import transforms
from torchdrug import data, core, layers, tasks, metrics, utils, models
from torchdrug.layers import functional
from torchdrug.core import Registry as R
import torch
from torch.utils import data as torch_data
from torch.nn import functional as F
import logging

from .tasks import NodePropertyPrediction
from .datasets import ATPBind
from .bert import BertWrapModel

class DisableLogger():
    def __enter__(self):
       logging.disable(logging.CRITICAL)
    def __exit__(self, exit_type, exit_value, exit_traceback):
       logging.disable(logging.NOTSET)
    

class Pipeline:
    possible_models = ['bert']
    possible_datasets = ['atpbind']
    
    
    def __init__(self, model, dataset, gpus, model_kwargs={}, task_kwargs={}):
        if model not in self.possible_models:
            raise ValueError('Model must be one of {}'.format(self.possible_models))
    
        if dataset not in self.possible_datasets:
            raise ValueError('Dataset must be one of {}'.format(self.possible_datasets))
                
        if model == 'bert':
            with DisableLogger():
                self.model = BertWrapModel(**model_kwargs)
        elif model == 'lm-gearnet':
            pass
            
        if dataset == 'atpbind':
            truncuate_transform = transforms.TruncateProtein(max_length=350, random=False)
            protein_view_transform = transforms.ProteinView(view='residue')
            transform = transforms.Compose([truncuate_transform, protein_view_transform])

            dataset = ATPBind(atom_feature=None, bond_feature=None,
                            residue_feature="default", transform=transform)

            self.train_set, self.valid_set, self.test_set = dataset.split()
            print("train samples: %d, valid samples: %d, test samples: %d" %
                (len(self.train_set), len(self.valid_set), len(self.test_set)))
            
        
        self.task = NodePropertyPrediction(
            self.model, 
            normalization=False,
            num_mlp_layer=2,
            metric=("micro_auroc", "mcc"),
            **task_kwargs
        )
        
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        with DisableLogger():
            self.solver = core.Engine(self.task,
                                        self.train_set,
                                        self.valid_set,
                                        self.test_set,
                                        optimizer,
                                        batch_size=1,
                                        log_interval=1000,
                                        gpus=gpus
            )
        
    def train(self, num_epoch):
        return self.solver.train(num_epoch=num_epoch)
    
    def evaluate(self):
        
        return self.solver.evaluate("test")
        
        
        
    
    
        
    