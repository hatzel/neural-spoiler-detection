import random

import torch
from tqdm import tqdm
from sklearn import metrics
from sklearn.model_selection import ParameterGrid
from torch.utils.data import DataLoader
import argparse

from datasets import PaddedBatch
from datasets import BinarySpoilerDataset
from pytorch_pretrained_bert import (
    BertTokenizer,
    BertForSequenceClassification
)
import util
from run import BertRun


def build_parser():
    parser = argparse.ArgumentParser(description="Spoiler Classificaiton")
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--train-data", required=True)
    parser.add_argument("--test-data", required=True)
    parser.add_argument("--name", help="Give this run a nice name.")
    subparsers = parser.add_subparsers(help="Grid search", dest="run_mode")
    grid_search = subparsers.add_parser("grid-search")
    single_run = subparsers.add_parser("single-run")
    single_run.add_argument("--mode", default="binary", choices=["binary"])
    single_run.add_argument("--batch-size", default=8, type=int)
    single_run.add_argument(
        "--learning-rate", default=(1 * 10 ** -5), type=float)
    single_run.add_argument("--epochs", default=3, type=int)
    return parser


def main(args):
    if args.run_mode == "grid-search":
        parameter_grid = list(ParameterGrid({
            "lr": [1 * 10 ** -5, 5 * 10 ** -5, 3 * 10 ** -5, 2 * 10 ** -5],
            "seed": [args.seed + n for n in range(3)],
            "num_epochs": [3, 4, 5],
        }))
        random.shuffle(parameter_grid)
        for params in parameter_grid:
            print("Using these parameters: ", params)
            run = BertRun.for_dataset(args.train_data, args.test_data)
            run.train(**params)
            result = run.test()
            result.save(args.name)
    elif args.run_mode == "single-run":
        run = BertRun.for_dataset(args.train_data, args.test_data)
        run.train(
            batch_size=args.batch_size,
            lr=args.learning_rate,
            num_epochs=args.epochs,
            seed=args.seed,
        )
        result = run.test()
        result.save(args.name)


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    main(args)
