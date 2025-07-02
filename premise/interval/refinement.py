import argparse
import logging
from typing import Optional

import numpy as np

from premise.interval.utils import setup_logging
from premise.interval.interval import Samples, Trace
from premise.interval.learning import (
    build_learning_params_args_parser,
    initial_interval_learning,
    interval_learning,
    premilinaries,
)
from premise.interval.loading import build_suo, build_suo_args_parser
from premise.interval.loss import distance_measures
from premise.system import SystemUnderObservation
from premise.interval.stopping_condition import (
    IntervalWidthCalculator,
    RefinementStoppingCondition,
    SampleCountStoppingCondition,
    StabilizationStoppingCondition,
    TargetDistanceCalculator,
    ThresholdStoppingCondition,
)
from premise.trace_generator import ConditionalIntervalTraceGenerator
from premise.interval.utils import logger


def refinement_learning(
    suo: SystemUnderObservation,
    existing_transitions: bool,
    learning_length: int,
    learning_amount: int,
    initial_learning_amount: int,
    refinement_stopping_condition: RefinementStoppingCondition,
    min_width: float,
    epsilon: float = 1 / 1000,
    i_i_nl: int = 5,
    i_i_nu: int = 10,
    i_nl: int = 10,
    i_nu: int = 20,
    intermediate_model_path: Optional[str] = None,
    conditional_sampling_type: str = "obs",
    verbose: int = 0,
):
    all_states, all_transitions, initial_states = suo.get_states_and_transitions(
        all_transitions=not existing_transitions
    )

    prefixes: list[Trace] = [tuple()]
    extra_samples: Samples = []

    initial_interval, strength_interval_initial, interval, strength_interval = (
        premilinaries(
            epsilon,
            i_i_nl,
            i_i_nu,
            i_nl,
            i_nu,
            all_states,
            all_transitions,
            initial_states,
        )
    )

    iteration = 0

    transitions_learned = []
    all_prefixes = []

    while True:
        iteration += 1
        if verbose > 0:
            logger.info(
                f"-----------------------\nRefinement iteration {iteration} with {len(prefixes)} prefixes"
            )
        if verbose > 1:
            logger.info(f"Prefixes: {prefixes}")

        all_prefixes.append(prefixes)

        amount = initial_learning_amount if prefixes == [tuple()] else learning_amount
        initial_samples = []
        samples = extra_samples
        for prefix in prefixes:
            if conditional_sampling_type == "obs":
                s = suo.generate_random_traces(
                    [s[1] for s in prefix],
                    learning_length,
                    amount,
                )
            elif conditional_sampling_type == "cond_state":
                if (
                    refinement_stopping_condition.distance_calculator.mon_comps
                    is not None
                ):
                    imc_ctr = ConditionalIntervalTraceGenerator(
                        refinement_stopping_condition.distance_calculator.mon_comps.ipomdp,
                        "target",
                    )
                else:
                    imc_ctr = None
                s = []
                for i in range(amount):
                    if (
                        imc_ctr is None
                        or refinement_stopping_condition.distance_calculator.mon_comps
                        is None
                        or len(prefix) == 0
                    ):
                        start_state = prefix[-1] if len(prefix) > 0 else None
                    else:
                        start_state_trace = imc_ctr.generate_random_trace(
                            [0]
                            + [
                                refinement_stopping_condition.distance_calculator.mon_comps.observation_map[
                                    s[1]
                                ]
                                for s in prefix
                            ],
                            len(prefix) + 1,
                        )[0]
                        start_state = next(
                            (
                                state
                                for state, index in refinement_stopping_condition.distance_calculator.mon_comps.state_index_map.items()
                                if index == start_state_trace[-1][0]
                            ),
                            None,
                        )

                    s.append(
                        suo.generate_random_traces(
                            [],
                            learning_length - len(prefix),
                            initial_state=start_state if len(prefix) > 0 else None,
                        )[0]
                    )
            elif conditional_sampling_type == "state":
                start_state = prefix[-1] if len(prefix) > 0 else None
                s = suo.generate_random_traces(
                    [],
                    learning_length - len(prefix),
                    initial_state=start_state if len(prefix) > 0 else None,
                    amount=amount,
                )
            else:
                raise ValueError(
                    f"Unknown conditional sampling type: {conditional_sampling_type}"
                )

            if len(prefix) == 0:
                initial_samples += s

            if conditional_sampling_type == "obs":
                samples += [t[len(prefix) :] for t in s]
            else:
                samples += s

        transitions_learned.append(sum(len(s) for s in samples))

        if len(initial_samples) > 0:
            initial_interval_learning(
                all_states,
                initial_samples,
                strength_interval_initial,
                initial_interval,
                min_width,
            )

        interval_learning(
            all_states,
            samples,
            interval,
            strength_interval,
            min_width,
            False,
        )

        if verbose > 0:
            logger.info(
                f"Finished learning with {transitions_learned[-1]} additional transitions (total: {suo.stats()['transition_count']})"
            )

        if intermediate_model_path is not None:
            save_imc(
                initial_interval,
                interval,
                suo,
                intermediate_model_path + "-" + str(iteration),
            )

        res = refinement_stopping_condition.check(interval, initial_interval)
        if res is not None:
            prefixes, extra_samples = res
        else:
            break

    stats = {"all_prefixes": all_prefixes, "transitions_learned": transitions_learned}

    return interval, initial_interval, stats


def save_imc(
    initial_interval,
    interval,
    suo: SystemUnderObservation,
    model_path: Optional[str] = None,
):
    if model_path is None:
        model_path = f"out/{suo.model_name}"
    np.save(f"{model_path}-initial_interval.npy", initial_interval)  # type: ignore
    np.save(f"{model_path}-interval.npy", interval)  # type: ignore


def ref_main(args: argparse.Namespace):
    setup_logging()

    suo, initial_amount, horizon = build_suo(args)
    if args.sample_length is None:
        if initial_amount is not None and horizon is not None:
            args.sample_length = initial_amount + horizon
        else:
            raise ValueError(
                "Either sample_length must be specified or initial_amount and horizon must be provided by the model."
            )
    if args.horizon is None:
        if horizon is not None:
            args.horizon = horizon
        else:
            raise ValueError(
                "Either horizon must be specified or it must be provided by the model."
            )
    if args.conformence_length is None:
        if initial_amount is not None:
            args.conformence_length = initial_amount
        else:
            raise ValueError(
                "Either conformence_length must be specified or initial_amount must be provided by the model."
            )

    distance = distance_measures[args.distance](args.distance_threshold)

    if args.distance_calculator == "target":
        distance_calculator = TargetDistanceCalculator(
            suo,
            args.horizon,
            distance,
            args.exact,
            args.precision,
            args.verbose,
        )
    elif args.distance_calculator == "interval":
        distance_calculator = IntervalWidthCalculator(
            suo,
            args.horizon,
            distance,
            args.exact,
            args.precision,
            args.verbose,
        )

    if args.stopping_criteria == "stabilization":
        ref_stop_cond = StabilizationStoppingCondition(
            suo,
            distance_calculator,
            args.stopping_deviation,
            args.stopping_patience,
            args.prefix_amount,
            args.verbose,
            args.conformence_length,
            args.conformence_amount,
        )
    elif args.stopping_criteria == "threshold":
        ref_stop_cond = ThresholdStoppingCondition(
            suo,
            distance_calculator,
            args.stopping_threshold,
            args.stopping_patience,
            args.prefix_amount,
            args.verbose,
            args.conformence_length,
            args.conformence_amount,
        )
    elif args.stopping_criteria == "samples":
        ref_stop_cond = SampleCountStoppingCondition(
            suo,
            distance_calculator,
            args.stopping_samples,
            args.refinement_amount,
            args.verbose,
            args.conformence_length,
            args.conformence_amount,
            args.sample_length,
            args.prefix_amount,
        )
    else:
        raise ValueError(f"Unknown stopping criteria: {args.stopping_criteria}")

    interval, initial_interval, ref_stats = refinement_learning(
        suo,
        args.existing_transitions,
        args.sample_length,
        args.refinement_amount,
        args.initial_amount,
        ref_stop_cond,
        args.interval_min_width,
        args.epsilon,
        args.initial_lower_strength,
        args.initial_upper_strength,
        args.trans_lower_strength,
        args.trans_upper_strength,
        args.model_path,
        args.conditional_sampling_type,
        args.verbose,
    )

    stats = ref_stop_cond.stats() | suo.stats() | ref_stats | {"args": vars(args)}
    if args.dump_stats:
        np.save(args.dump_stats, stats)  # type: ignore

    save_imc(initial_interval, interval, suo, args.model_path)
    return stats


def ref_args_parser():

    parser = argparse.ArgumentParser(description="Conformance checking")

    build_suo_args_parser(parser)

    learning_group = parser.add_argument_group("Learning")
    learning_group.add_argument(
        "-ll",
        "--sample-length",
        type=int,
        help="Length of the samples to generate for learning",
    )
    learning_group.add_argument(
        "-ia",
        "--initial-amount",
        type=int,
        default=100,
        help="Amount of initial samples to learn on",
    )
    learning_group.add_argument(
        "-ra",
        "--refinement-amount",
        type=int,
        default=2,
        help="Amount of refinement samples to learn on per prefix",
    )
    learning_group.add_argument(
        "-pa",
        "--prefix-amount",
        type=int,
        default=10,
        help="Amount of prefixes per trace. Should not be larger then the conformance length",
    )
    learning_group.add_argument(
        "-cst",
        "--conditional-sampling-type",
        choices=["obs", "state", "cond_state"],
        default="cond_state",
        help="Type of conditional sampling to use, 'obs' conditions on the trace, 'state' starts in the final state of the condition",
    )
    learning_group.add_argument(
        "-m", "--model-path", type=str, default=None, help="Path to store the model"
    )
    trans_del_group = learning_group.add_mutually_exclusive_group(required=False)
    trans_del_group.add_argument(
        "-t",
        "--existing-transitions",
        action="store_true",
        help="Use only real transitions",
    )
    trans_del_group.add_argument(
        "--min-trans-prob",
        type=float,
        default=0.01,
        help="Minimum transition probability assumed of a transition",
    )

    conformence_group = parser.add_argument_group("Conformance")
    conformence_group.add_argument(
        "-e",
        "--exact",
        default=True,
        action="store_true",
        help="Use exact conformance checking",
    )
    conformence_group.add_argument(
        "--no-exact",
        dest="exact",
        action="store_false",
        help="Do not use exact conformance checking",
    )
    conformence_group.add_argument(
        "-p",
        "--precision",
        type=float,
        default=1e-6,
        help="Precision to use for the exact conformance checking",
    )
    conformence_group.add_argument(
        "-d",
        "--distance",
        choices=distance_measures.keys(),
        default="umse",
        help="Distance measure to use",
    )
    conformence_group.add_argument(
        "-dt",
        "--distance-threshold",
        type=float,
        help="Distance threshold to use for the threshold distance",
    )
    conformence_group.add_argument(
        "-ca",
        "--conformence-amount",
        type=int,
        default=100,
        help="Amount of samples to test on in each refimement iteration",
    )
    conformence_group.add_argument(
        "-cl",
        "--conformence-length",
        type=int,
        default=None,
        help="Length of the samples to generate for conformance checking. Defaults to the sample length",
    )
    conformence_group.add_argument(
        "-ho", "--horizon", type=int, help="The horizon to monitor on"
    )
    conformence_group.add_argument(
        "-sc",
        "--stopping-criteria",
        default="threshold",
        choices=["threshold", "stabilization", "samples"],
    )
    conformence_group.add_argument(
        "-st",
        "--stopping-threshold",
        type=float,
        default=0.01,
        help="The threshold to stop refinement at. The distance must be below this threshold",
    )
    conformence_group.add_argument(
        "-sd",
        "--stopping-deviation",
        type=float,
        default=0.05,
        help="The relative deviation to stop refinement at",
    )
    conformence_group.add_argument(
        "-sp",
        "--stopping-patience",
        type=int,
        default=3,
        help="The amount of iterations to wait before stopping refinement",
    )
    conformence_group.add_argument(
        "-ss",
        "--stopping-samples",
        type=int,
        help="The amount of samples to stop refinement at",
    )
    conformence_group.add_argument(
        "-dc",
        "--distance-calculator",
        choices=["target", "interval"],
        default="interval",
    )

    parser.add_argument(
        "--dump-stats",
        type=str,
        help="Path to the file to dump stats to",
    )

    parser.add_argument(
        "-ri",
        "--run-id",
        type=int,
        default=0,
        help="Run ID to use for the experiment. Used to distinguish between different runs in the same model path.",
    )

    parser.add_argument("--verbose", "-v", action="count", default=0)

    build_learning_params_args_parser(parser)
    return parser


if __name__ == "__main__":
    parser = ref_args_parser()
    parsed_args = parser.parse_args()

    ref_main(parsed_args)
