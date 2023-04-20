from torchdrug.tasks import NodePropertyPrediction
from torchdrug import transforms
from torchdrug import data, core, layers, tasks, metrics, utils, models
from torchdrug.layers import functional
from torchdrug.core import Registry as R
from torchdrug.models import GearNet

import torch
from torch.utils import data as torch_data
from torch.nn import functional as F

from lib.datasets import ATPBind

truncuate_transform = transforms.TruncateProtein(max_length=350, random=False)
protein_view_transform = transforms.ProteinView(view='residue')
#transform = transforms.Compose([truncuate_transform, protein_view_transform])
transform = transforms.Compose([protein_view_transform])

dataset = ATPBind(atom_feature=None, bond_feature=None,
                  residue_feature="default", transform=transform)

train_set, valid_set, test_set = dataset.split()
print("train samples: %d, valid samples: %d, test samples: %d" %
      (len(train_set), len(valid_set), len(test_set)))

#from lib.tasks import NodePropertyPrediction

model = GearNet(
    num_relation=4,
    input_dim=21,
    hidden_dims=[1024, 1024],
)


task = NodePropertyPrediction(model, normalization=False, num_mlp_layer=2, metric=(
    "micro_auroc", "micro_auprc", "macro_auprc", "macro_auroc"))
optimizer = torch.optim.Adam(task.parameters(), lr=1e-4)
solver = core.Engine(task, train_set, valid_set, test_set,
                     optimizer, batch_size=4, log_interval=10000000, gpus=[2])
solver.train(num_epoch=100)
solver.evaluate("valid")
solver.evaluate("test")
