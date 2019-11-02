import random

from sklearn.model_selection import ParameterGrid
import argparse
from torch.utils.tensorboard import SummaryWriter

import util
from run import BertRun


def build_parser():
    parser = argparse.ArgumentParser(description="Spoiler Classificaiton")
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--train-data", required=False, help="Required for training but not for testing.", nargs="*")
    parser.add_argument("--test-data", required=True, nargs="*")
    parser.add_argument("--test-loss-report", action="store_true")
    parser.add_argument("--test-loss-early-stopping", action="store_true")
    parser.add_argument(
        "--results-file",
        help="Store prediction results to a file (only in test mode for now).",
        default="results.log"
    )
    parser.add_argument("--token-based", action="store_true")
    parser.add_argument("--name", help="Give this run a nice name.")
    parser.add_argument("--base-model", help="Which BERT model to perform fine tuning on.", default="bert-base-cased")
    parser.add_argument("--logdir", help="Tensorboard log directory")
    parser.add_argument("--limit-train", help="Limit training dataset to a specifc number of samples", type=int, default=None)
    parser.add_argument("--limit-test", help="Limit test dataset to a specifc number of samples", type=int, default=None)
    subparsers = parser.add_subparsers(help="Grid search", dest="run_mode")
    grid_search = subparsers.add_parser("grid-search")
    test_mode = subparsers.add_parser("test", help="Test an existing model.")
    test_mode.add_argument("model", help="Model to test against.")
    single_run = subparsers.add_parser("single-run")
    single_run.add_argument("--scheduler-epochs", type=int, help="How many epochs to base learning rate schedule on.")
    single_run.add_argument("--mode", default="binary", choices=["binary"])
    single_run.add_argument("--batch-size", default=8, type=int)
    single_run.add_argument(
        "--learning-rate", default=(1 * 10 ** -5), type=float)
    single_run.add_argument("--epochs", default=3, type=int)
    return parser


def main(args):
    if args.run_mode != "test" and not args.train_data:
        raise Exception("When training make sure to supply training data with --train-data!")
    if args.limit_train or args.limit_test:
        print("Warning: You supplied a limit, we will only take the first n samples in the file, no full shuffle is performed.")
    print(f"Token based: {args.token_based}")
    if args.run_mode == "grid-search":
        parameter_grid = list(ParameterGrid({
            "lr": [1 * 10 ** -6,
                   5 * 10 ** -7,
                   5 * 10 ** -7,
                   5 * 10 ** -6,
                   1 * 10 ** -5,
                   2 * 10 ** -5,
                   3 * 10 ** -5,
                   5 * 10 ** -5,
                   10 ** -4],
            "seed": [args.seed], # Let's not optimize for this, it takes a long time
            "num_epochs": [4, 12], # We can just use early stopping to explore most of this
        }))
        random.shuffle(parameter_grid)
        model = None
        optimizer = None
        for params in parameter_grid:
            writer = SummaryWriter(args.logdir)
            print("Using these parameters: ", params)
            run = BertRun.for_dataset(
                args.train_data,
                args.test_data,
                args.base_model,
                train_limit=args.limit_train,
                limit_test=args.limit_test,
                test_loss_early_stopping=args.test_loss_early_stopping,
                test_loss_report=args.test_loss_report,
                model=model,
                optimizer=optimizer,
            )
            run.train(writer=writer, **params)
            util.seed_for_testing()
            result = run.test(writer=writer)
            result.save(args.name, writer=writer)
            # We need to preserve the same optimizer instance to avoid initalizing amp multiple times
            optimizer = run.optimizer
            model = run.classifier
    elif args.run_mode == "single-run":
        writer = SummaryWriter(args.logdir)
        run = BertRun.for_dataset(
            args.train_data,
            args.test_data,
            args.base_model,
            limit_test=args.limit_test,
            train_limit=args.limit_train,
            token_based=args.token_based,
            test_loss_early_stopping=args.test_loss_early_stopping,
            test_loss_report=args.test_loss_report,
            scheduler_epochs=args.scheduler_epochs
        )
        run.train(
            writer=writer,
            batch_size=args.batch_size,
            lr=args.learning_rate,
            num_epochs=args.epochs,
            seed=args.seed,
        )
        util.seed_for_testing()
        result = run.test(writer=writer)
        result.save(args.name, writer=writer)
    elif args.run_mode == "test":
        run = BertRun.from_file(
            args.model,
            None,
            args.test_data,
            args.base_model,
            limit_test=args.limit_test,
            token_based=args.token_based,
        )
        util.seed_for_testing()
        run.test(results_file_name=args.results_file)


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    main(args)
