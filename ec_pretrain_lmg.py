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
    parser.add_argument("--hidden_dim_count", type=int, default=6,
                        help="Number of hidden dimensions")
    parser.add_argument("--hidden_dim_size", type=int, default=512,
                        help="Size of each hidden dimension")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Batch size for training")
    return parser.parse_args()


def main():
    # https://wsshin.tistory.com/12
    os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    args = parse_args()

    # Load dataset
    truncuate_transform = transforms.TruncateProtein(
        max_length=350, random=False)
    protein_view_transform = transforms.ProteinView(view='residue')
    transform = transforms.Compose(
        [truncuate_transform, protein_view_transform])

    dataset = datasets.EnzymeCommission(
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
                                                                     geometry.SequentialEdge(max_distance=2)],
                                                        edge_feature="gearnet")

    lm_gearnet = LMGearNetModel(args.gpu,
                                gearnet_hidden_dim_size=args.hidden_dim_size, 
                                gearnet_hidden_dim_count=args.hidden_dim_count
                                )

    task = tasks.AttributeMasking(lm_gearnet, graph_construction_model=graph_construction_model,
                                  mask_rate=0.15, num_mlp_layer=2)

    optimizer = torch.optim.Adam(task.parameters(), lr=1e-4)
    with DisableLogger():
        solver = core.Engine(task, train_set, valid_set, test_set, optimizer,
                             gpus=[args.gpu], batch_size=args.batch_size)
    print("Building model and solver done")

    if args.pretrained is not None:
        # Load pretrained model
        solver.load(args.pretrained)

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

    torch.save(lm_gearnet.gearnet.state_dict(), "ResidueType_lmg_%d_%d_%.5f.pth" %
                (args.hidden_dim_count,
                 args.hidden_dim_size,
                 result['accuracy'].item()))


if __name__ == "__main__":
    main()
