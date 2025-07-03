import argparse
import numpy as np
import os

from premise.interval.loading import build_suo, build_suo_args_parser


def build_learning_args_parser(parser: argparse.ArgumentParser):
    group = parser.add_argument_group("Learning Parameters")
    group.add_argument(
        "-m", "--model-path", type=str, default=None, help="Path to store the model"
    )
    group.add_argument(
        "-a", "--amount", type=int, default=1000, help="Amount of samples to generate"
    )
    group.add_argument(
        "-l", "--length", type=int, default=20, help="Length of the samples to generate"
    )
    group.add_argument(
        "-t",
        "--existing-transitions",
        action="store_true",
        help="Use only real transitions as defined by the model",
    )
    group.add_argument(
        "--min-trans-prob",
        type=float,
        default=None,
        help="Minimum transition probability assumed of a transition",
    )


def build_learning_params_args_parser(parser: argparse.ArgumentParser):
    param_group = parser.add_argument_group("Internal Learning Parameters")
    param_group.add_argument(
        "--epsilon",
        type=float,
        default=1 / 1000,
        help="Epsilon value for the initial interval",
    )
    param_group.add_argument(
        "--initial-lower-strength",
        type=int,
        default=5,
        help="Initial lower bound of strength interval for initial distribution",
    )
    param_group.add_argument(
        "--initial-upper-strength",
        type=int,
        default=10,
        help="Initial upper bound of strength interval for initial distribution",
    )
    param_group.add_argument(
        "--trans-lower-strength",
        type=int,
        default=10,
        help="Initial lower bound of strength interval",
    )
    param_group.add_argument(
        "--trans-upper-strength",
        type=int,
        default=20,
        help="Initial upper bound of strength interval",
    )
    param_group.add_argument(
        "--interval-min-width",
        type=float,
        default=0.0,
        help="Fix minimum interval width",
    )


parser = argparse.ArgumentParser(description="Learn an IMC")
build_suo_args_parser(parser)
build_learning_args_parser(parser)
build_learning_params_args_parser(parser)
parser.add_argument(
    "-v",
    "--verbose",
    action="count",
    default=0,
    help="Increase verbosity level (can be used multiple times)",
)

args = parser.parse_args()

suo = build_suo(args)

all_states, all_transitions, initial_states = suo.get_states_and_transitions(
    all_transitions=not args.existing_transitions
)

print(len(all_states))
