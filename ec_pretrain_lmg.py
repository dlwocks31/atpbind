import argparse

from torchdrug import datasets, transforms
from torchdrug import layers, tasks, core, models
from torchdrug.layers import geometry
import torch
from lib.disable_logger import DisableLogger
from lib.custom_models import LMGearNetModel

import os
from time import sleep
import traceback


def parse_args():
    parser = argparse.ArgumentParser(description="Script options")
    parser.add_argument("--pretrained", type=str, required=False,
                        help="Path to the pretrained weight file")
    parser.add_argument("--gpu", type=int, default=0,
                        help="GPU to use")
    parser.add_argument("--bert_freeze_layer_count", type=int, default=30,
                        help="Number of layers to freeze in BERT")
    parser.add_argument("--hidden_dim_count", type=int, default=6,
                        help="Number of hidden dimensions")
    parser.add_argument("--hidden_dim_size", type=int, default=512,
                        help="Size of each hidden dimension")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Batch size for training")
    parser.add_argument("--sequential_max_distance", type=int, default=2,
                        help="Max distance for sequential edge")
    parser.add_argument("--lr", type=float, default=1e-5,
                        help="Learning rate")
    parser.add_argument("--dataset", type=str, default="ec",
                        help="Dataset to use")
    return parser.parse_args()


def main():
    # # https://wsshin.tistory.com/12
    # os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    args = parse_args()

    # Load dataset
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

    # Build Model and Solver
    graph_construction_model = layers.GraphConstruction(node_layers=[geometry.AlphaCarbonNode()],
                                                        edge_layers=[geometry.SpatialEdge(radius=10.0, min_distance=5),
                                                                     geometry.KNNEdge(
                                                                         k=10, min_distance=5),
                                                                     geometry.SequentialEdge(max_distance=args.sequential_max_distance)],
                                                        edge_feature="gearnet")

    lm_gearnet = LMGearNetModel(args.gpu,
                                gearnet_hidden_dim_size=args.hidden_dim_size, 
                                gearnet_hidden_dim_count=args.hidden_dim_count,
                                bert_freeze=args.bert_freeze_layer_count==30,
                                bert_freeze_layer_count=args.bert_freeze_layer_count,
                                graph_sequential_max_distance=args.sequential_max_distance,
                                )

    task = tasks.AttributeMasking(lm_gearnet, graph_construction_model=graph_construction_model,
                                  mask_rate=0.15, num_mlp_layer=2)

    optimizer = torch.optim.Adam(task.parameters(), lr=args.lr)
    with DisableLogger():
        solver = core.Engine(task, train_set, valid_set, test_set, optimizer,
                             gpus=[args.gpu], batch_size=args.batch_size)
    print("Building model and solver done")

    if args.pretrained is not None:
        # Load pretrained model
        state_dict = torch.load(args.pretrained, map_location=f'cuda:{args.gpu}')
        lm_gearnet.load_state_dict(state_dict)

    while True:
        try:
            solver.train(num_epoch=1)
        except (RuntimeError, IndexError) as e:
            print(e)
            print("RuntimeError occurred. Continue training")
            sleep(60)

        validate_and_save(solver, lm_gearnet, args)


def validate_and_save(solver, lm_gearnet, args):
    fail_cnt = 0
    while True:
        try:
            result = solver.evaluate("valid")
            break
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
    if accuracy < result['accuracy'].item():
        print(f"Saving weight with accuracy {result['accuracy']:.5f}")
        torch.save(lm_gearnet.state_dict(), f"{name_prefix}_{result['accuracy']:.5f}.pth")
        for file in files:
            os.remove(file)

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
