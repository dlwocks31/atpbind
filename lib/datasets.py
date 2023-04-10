import torch
from torch.utils import data as torch_data
from torchdrug import data
import os


class ATPBind(data.ProteinDataset):
    splits = ["train", "valid", "test"]
    target_fields = ['binding']

    def __init__(self, path=None, verbose=1, valid_ratio=0.1, **kwargs):
        if path is None:
            path = os.path.dirname(__file__)
        self.num_samples = []
        self.valid_ratio = valid_ratio
        sequences, targets = self.get_seq_target(path)
        self.load_sequence(sequences, targets)
        # [20,21,41]#[350, 38, 41]

    def get_seq_target(self, path):
        sequences, targets = [], []

        with open(os.path.join(path, 'train.txt')) as f:
            lines = f.readlines()
            self.num_samples.append(len(lines))
            for line in lines:
                sequence = line.split(' : ')[-1].strip()
                sequences.append(sequence)

                target = line.split(' : ')[-2].split(' ')
                target_indices = []
                for index in target:
                    target_indices.append(int(index[1:]))
                target = []
                for index in range(len(sequence)):
                    if index+1 in target_indices:
                        target.append(1)
                    else:
                        target.append(0)
                targets.append(target)

        with open(os.path.join(path, 'test.txt')) as f:
            lines = f.readlines()
            self.num_samples.append(len(lines))
            for line in lines:
                sequence = line.split(' : ')[-1].strip()
                sequences.append(sequence)

                target = line.split(' : ')[-2].split(' ')
                target_indices = []
                for index in target:
                    target_indices.append(int(index[1:]))
                target = []
                for index in range(len(sequence)):
                    if index+1 in target_indices:
                        target.append(1)
                    else:
                        target.append(0)
                targets.append(target)

        # calculate set lengths
        total_samples = sum(self.num_samples)
        val_num = int(total_samples*self.valid_ratio)
        self.num_samples = [self.num_samples[0] -
                            val_num, val_num, self.num_samples[1]]
        print('Split num: ', self.num_samples)

        targets_ = {"binding": targets}
        return sequences, targets_

    def get_item(self, index):
        if self.lazy:
            graph = data.Protein.from_sequence(
                self.sequences[index], **self.kwargs)
        else:
            graph = self.data[index]
        with graph.residue():
            target = torch.as_tensor(
                self.targets["binding"][index], dtype=torch.long).view(-1, 1)
            graph.target = target
            mask = torch.ones_like(target).bool()
            graph.mask = mask
        graph.view = 'residue'
        item = {"graph": graph}
        if self.transform:
            item = self.transform(item)
        return item

    def split(self, keys=None):
        keys = keys or self.splits
        offset = 0
        splits = []
        for split_name, num_sample in zip(self.splits, self.num_samples):
            if split_name in keys:
                split = torch_data.Subset(
                    self, range(offset, offset + num_sample))
                splits.append(split)
            offset += num_sample
        return splits
