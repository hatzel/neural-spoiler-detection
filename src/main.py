import random

from sklearn.model_selection import ParameterGrid
import argparse
from torch.utils.tensorboard import SummaryWriter

import util
from run import BertRun


def build_parser():
    parser = argparse.ArgumentParser(description="Spoiler Classificaiton")
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--train-data", required=True)
    parser.add_argument("--test-data", required=True)
    parser.add_argument("--token-based", action="store_true")
    parser.add_argument("--name", help="Give this run a nice name.")
    parser.add_argument("--base-model", help="Which BERT model to perform fine tuning on.", default="bert-base-cased")
    parser.add_argument("--limit", help="Limit test and train dataset to a specifc number of samples", type=int, default=None)
    subparsers = parser.add_subparsers(help="Grid search", dest="run_mode")
    grid_search = subparsers.add_parser("grid-search")
    test_mode = subparsers.add_parser("test", help="Test an existing model.")
    test_mode.add_argument("model", help="Model to test against.")
    single_run = subparsers.add_parser("single-run")
    single_run.add_argument("--mode", default="binary", choices=["binary"])
    single_run.add_argument("--batch-size", default=8, type=int)
    single_run.add_argument(
        "--learning-rate", default=(1 * 10 ** -5), type=float)
    single_run.add_argument("--epochs", default=3, type=int)
    return parser


def main(args):
    print(f"Token based: {args.token_based}")
    writer = SummaryWriter()
    if args.run_mode == "grid-search":
        parameter_grid = list(ParameterGrid({
            "lr": [1 * 10 ** -5, 5 * 10 ** -5, 3 * 10 ** -5, 2 * 10 ** -5],
            "seed": [args.seed + n for n in range(3)],
            "num_epochs": [3, 4, 5],
        }))
        random.shuffle(parameter_grid)
        for params in parameter_grid:
            print("Using these parameters: ", params)
            run = BertRun.for_dataset(args.train_data, args.test_data, args.base_model)
            run.train(writer=writer, **params)
            util.seed(1)
            result = run.test(writer=writer)
            result.save(args.name)
    elif args.run_mode == "single-run":
        run = BertRun.for_dataset(args.train_data, args.test_data, args.base_model, limit=args.limit, token_based=args.token_based)
        run.train(
            writer=writer,
            batch_size=args.batch_size,
            lr=args.learning_rate,
            num_epochs=args.epochs,
            seed=args.seed,
        )
        util.seed(1)
        result = run.test(writer=writer)
        result.save(args.name)
    elif args.run_mode == "test":
        run = BertRun.from_file(args.model, args.train_data, args.test_data, limit=args.limit)
        util.seed(1)
        run.test()


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    main(args)
