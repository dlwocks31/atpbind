import torch
from torch.utils import data as torch_data
from torchdrug import data
import os
from collections import defaultdict
import numpy as np


class ATPBind(data.ProteinDataset):
    splits = ["train", "valid", "test"]
    target_fields = ['binding']
    deny_list = []

    def __init__(self, path=None, limit=-1, verbose=1, valid_ratio=0.1, **kwargs):
        if path is None:
            path = os.path.dirname(__file__)
        self.num_samples = []
        self.valid_ratio = valid_ratio
        sequences, targets, _ = self.get_seq_target(path, limit)
        self.load_sequence(sequences, targets, **kwargs)
        # [20,21,41]#[350, 38, 41]

    def get_seq_target(self, path, limit):
        sequences, targets, pdb_ids = [], [], []

        for file in ['train.txt', 'test.txt']:
            num_samples, seq, tgt, ids = read_file(os.path.join(path, file))
            # Filter sequences, targets, and pdb_ids based on deny_list
            filtered_seq, filtered_tgt, filtered_ids = zip(
                *[(s, t, i) for s, t, i in zip(seq, tgt, ids) if i not in self.deny_list])

            if limit > 0:
                filtered_seq = filtered_seq[:limit]
                filtered_tgt = filtered_tgt[:limit]
                filtered_ids = filtered_ids[:limit]

            sequences += filtered_seq
            targets += filtered_tgt
            pdb_ids += filtered_ids

            self.num_samples.append(len(filtered_seq))

        # calculate set lengths
        total_samples = sum(self.num_samples)
        val_num = int(total_samples*self.valid_ratio)
        self.num_samples = [self.num_samples[0] -
                            val_num, val_num, self.num_samples[1]]
        print('Split num: ', self.num_samples)

        targets_ = {"binding": targets}
        return sequences, targets_, pdb_ids

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


def read_file(path):
    '''
    Read from ATPBind dataset.
    '''
    sequences, targets, pdb_ids = [], [], []
    with open(path) as f:
        lines = f.readlines()
        num_samples = len(lines)
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

            pdb_id = line.split(' : ')[0]
            pdb_ids.append(pdb_id)
    return num_samples, sequences, targets, pdb_ids


class ATPBind3D(data.ProteinDataset):
    splits = ["train", "valid", "test"]
    target_fields = ['binding']
    # see `generate_pdb.py`
    # also, 4TU0A includes two alpha carbon that would not be parsed with torchprotein (number 7 / 128)
    # This is probably because non-standard pdb file, or non-standard torchprotein parser
    deny_list = ['3CRCA', '2C7EG', '3J2TB', '3VNUA',
                 '4QREA', '5J1SB', '1MABB', '3LEVH', '3BG5A',
                 '4TU0A',
                 ]

    fold_count = 5
    
    def __init__(self, path=None, limit=-1, **kwargs):
        if path is None:
            path = os.path.dirname(__file__)
        self.num_samples = []
        _, targets, pdb_ids = self.get_seq_target(path, limit)
        pdb_files = [os.path.join(path, '../data/pdb/%s.pdb' % pdb_id)
                     for pdb_id in pdb_ids if pdb_id not in self.deny_list]

        self.load_pdbs(pdb_files, **kwargs)
        self.targets = defaultdict(list)
        self.targets["binding"] = targets["binding"]

        self.fold_ranges = np.array_split(np.arange(self.train_sample_count), self.fold_count)

        

    def initialize_undersampling(self, masks=None):
        if masks is not None:
            print('Initialize Undersampling: fixed mask')
            self.masks = masks
        else:
            print('Initialize Undersampling: all ones')
            self.masks = [
                torch.ones(len(target)).bool()
                for target in self.targets["binding"]
            ]
        return self
        

    def get_seq_target(self, path, limit):
        sequences, targets, pdb_ids = [], [], []

        for file in ['train.txt', 'test.txt']:
            num_samples, seq, tgt, ids = read_file(os.path.join(path, file))
            # Filter sequences, targets, and pdb_ids based on deny_list
            filtered_seq, filtered_tgt, filtered_ids = zip(
                *[(s, t, i) for s, t, i in zip(seq, tgt, ids) if i not in self.deny_list])

            if limit > 0:
                filtered_seq = filtered_seq[:limit]
                filtered_tgt = filtered_tgt[:limit]
                filtered_ids = filtered_ids[:limit]

            sequences += filtered_seq
            targets += filtered_tgt
            pdb_ids += filtered_ids

            if file == 'train.txt':
                self.train_sample_count = len(filtered_seq)
            elif file == 'test.txt':
                self.test_sample_count = len(filtered_seq)
            else:
                raise NotImplementedError

        targets_ = {"binding": targets}
        return sequences, targets_, pdb_ids


    def _is_train_set(self, index):
        return (index < self.train_sample_count) and (index not in self.fold_ranges[self.valid_fold_num])

    def _generate_mask(self, index):
        if not self._is_train_set(index) or self.masks is None:
            # if not train set, do not mask!
            return torch.ones(len(self.targets["binding"][index])).bool()
        return self.masks[index]
    
    def valid_fold(self):
        return self.fold_ranges[self.valid_fold_num]

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
            graph.mask = self._generate_mask(index).view(-1, 1)
        graph.view = 'residue'
        item = {"graph": graph}
        if self.transform:
            item = self.transform(item)
        # print(f'get_item {index}, mask {item["graph"].mask.sum()} / {len(item["graph"].mask)}')
        return item

    def split(self, valid_fold_num=0):
        assert(valid_fold_num < self.fold_count and valid_fold_num >= 0)

        self.valid_fold_num = valid_fold_num

        splits = [
            torch_data.Subset(self, to_int_list(
                np.concatenate(self.fold_ranges[:valid_fold_num] + self.fold_ranges[valid_fold_num+1:])
            )), # train
            torch_data.Subset(self, to_int_list(self.fold_ranges[valid_fold_num])), # valid
            torch_data.Subset(self, list(range(self.train_sample_count, self.train_sample_count + self.test_sample_count))), # test
        ]
        return splits

def to_int_list(np_arr):
    return [int(i) for i in np_arr]