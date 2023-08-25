import argparse

from torchdrug import datasets, transforms
from torchdrug import layers, tasks, core, models
from torchdrug.layers import geometry
from torch.utils import data as torch_data
import torch
from lib.disable_logger import DisableLogger
from lib.custom_models import LMGearNetModel
from lib.pretrain import CustomAttributeMasking
from lib.utils import dict_tensor_to_num

import os
from time import sleep
from itertools import count
import traceback
import pandas as pd

def read_initial_csv(path):
    try:
        return pd.read_csv(path)
    except (FileNotFoundError, IndexError):
        # File does not exist, or it is empty
        return pd.DataFrame()

def parse_args():
    parser = argparse.ArgumentParser(description="Script options")
    parser.add_argument("--pretrained", type=str, required=False,
                        help="Path to the pretrained weight file")
    parser.add_argument("--gpu", type=int, default=0,
                        help="GPU to use")
    parser.add_argument("--hidden_dim_count", type=int, default=4,
                        help="Number of hidden dimensions")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Batch size for training")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate")
    parser.add_argument("--dataset", type=str, default="ec",
                        help="Dataset to use")
    parser.add_argument("--task", type=str, default="rtp")
    parser.add_argument("--lm", type=str, default="esm-t33",
                        help="Language model to use")
    return parser.parse_args()

def get_last_ce(meter):
    from statistics import mean
    index = slice(meter.epoch2batch[-2], meter.epoch2batch[-1])
    ce_records = meter.records['cross entropy'][index]
    return mean(ce_records)

def main():
    # # https://wsshin.tistory.com/12
    # os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    args = parse_args()
    print(args)
    print(type(args))

    # Build Model and Solver
    print('Build Model And Solver')

    graph_construction_model = layers.GraphConstruction(node_layers=[geometry.AlphaCarbonNode()],
                                                        edge_layers=[geometry.SpatialEdge(radius=10.0, min_distance=5),
                                                                     geometry.KNNEdge(
                                                                         k=10, min_distance=5),
                                                                     geometry.SequentialEdge(max_distance=2)],
                                                        edge_feature="gearnet")

    lm_gearnet = LMGearNetModel(args.gpu, lm_type=args.lm, gearnet_hidden_dim_count=args.hidden_dim_count)
    
    lm_gearnet.freeze_lm(freeze_all=True)

    if args.task == 'rtp':
        task = CustomAttributeMasking(lm_gearnet, graph_construction_model=graph_construction_model,
                                    mask_rate=0.15, num_mlp_layer=2)
    elif args.task == 'mvcl':
        model = models.MultiviewContrast(lm_gearnet, noise_funcs=[geometry.IdentityNode(), geometry.RandomEdgeMask(mask_rate=0.15)],
                                        crop_funcs=[geometry.SubsequenceNode(max_length=50), 
                                                    geometry.SubspaceNode(entity_level="residue", min_neighbor=15, min_radius=15.0)], num_mlp_layer=2)
        task = tasks.Unsupervised(model, graph_construction_model=graph_construction_model)

    optimizer = torch.optim.AdamW(task.parameters(), lr=args.lr, weight_decay=1e-4)
    # T_0 = 50
    # scheduler = CosineAnnealingWarmRestarts(optimizer, T_0, eta_min=1e-6)
    scheduler = None
    

    
    # Load dataset
    print('Load Dataset')
    truncuate_transform = transforms.TruncateProtein(
        max_length=350, random=False)
    protein_view_transform = transforms.ProteinView(view='residue')
    transform = transforms.Compose(
        [truncuate_transform, protein_view_transform])

    if args.dataset == 'ec':
        dataset = datasets.EnzymeCommission(
            "~/protein-datasets/", transform=transform, atom_feature=None, bond_feature=None)
    elif args.dataset == 'go':
        dataset = datasets.GeneOntology(
            "~/protein-datasets/", transform=transform, atom_feature=None, bond_feature=None)
    train_set, valid_set, test_set = dataset.split()
        
    print(dataset)
    print("train samples: %d, valid samples: %d, test samples: %d" %
          (len(train_set), len(valid_set), len(test_set)))

    print('Build Solver')
    with DisableLogger():
        solver = core.Engine(task, train_set, valid_set, test_set, optimizer, scheduler=scheduler,
                             gpus=[args.gpu], batch_size=args.batch_size)
    
    print("Building model and solver done")

    if args.pretrained is not None:
        # Load pretrained model
        state_dict = torch.load(args.pretrained, map_location=f'cuda:{args.gpu}')
        lm_gearnet.gearnet.load_state_dict(state_dict)

    for epoch in count(start=1):
        try:
            solver.train(num_epoch=1)
        except (RuntimeError, IndexError) as e:
            print(e)
            print("RuntimeError occurred. Continue training")
            sleep(60)

        if args.task == 'rtp':
            validate_and_save(solver, lm_gearnet, args, epoch, cur_lr=args.lr)
        elif args.task == 'mvcl':
            torch.save(lm_gearnet.state_dict(), "mvcl.pth")


def validate_and_save(solver, lm_gearnet, args, epoch, cur_lr):
    CSV_FILE = f"pretrain_{args.dataset}_{args.task}_{args.hidden_dim_count}.csv"
    fail_cnt = 0
    while True:
        try:
            result = solver.evaluate("valid")
            result = dict_tensor_to_num(result)
            break # evaluate 성공하면 나와서 save 시도
        except (RuntimeError, IndexError) as e:
            print(e)
            fail_cnt += 1
            if fail_cnt == 10:
                print("Too many errors. Stop validating")
                exit(0)
            print("Detailed stack trace:")
            traceback.print_exc()  # print detailed stack trace
            print(
                f"RuntimeError occurred: fail_cnt = {fail_cnt}. Continue validating")
            sleep(60)

    name_prefix = f"lmg_{args.dataset}_{args.hidden_dim_count}"
    files, accuracy = parse_current_saved_weight(name_prefix)
    if accuracy < result['accuracy']:
        print(f"Saving weight with accuracy {result['accuracy']:.5f}")
        torch.save(lm_gearnet.gearnet.state_dict(), f"{name_prefix}_{result['accuracy']:.5f}.pth")
        for file in files:
            os.remove(file)
    
    # save train record to csv
    df = read_initial_csv(CSV_FILE)
    new_row = pd.DataFrame.from_dict([{
        'epoch': epoch,
        'time': str(pd.Timestamp.now()),
        'cur_lr': cur_lr,
        'train_loss': get_last_ce(solver.meter),
        'valid_loss': result['cross entropy'],
        'valid_accuracy': result['accuracy'],
    }])
    df = pd.concat([df, new_row])
    df.to_csv(CSV_FILE, index=False)

def find_files_with_prefix(prefix):
    import re
    current_dir = os.getcwd()
    files_with_prefix = []
    for filename in os.listdir(current_dir):
        match_obj = re.match(f'{prefix}_(0\.\d*)\.pth', filename)
        if match_obj:
            files_with_prefix.append(filename)
    return files_with_prefix

def parse_current_saved_weight(prefix):
    # take all files with prefix
    files = find_files_with_prefix(prefix)
    accuracies = [float(i.replace('.pth', '').split('_')[-1]) for i in files]
    print(accuracies)
    return files, (max(accuracies) if len(accuracies) > 0 else 0)

if __name__ == "__main__":
    main()
