from abc import ABC
import argparse
from typing import Any

import numpy as np

from premise.interval.conformence import test_monitor
from premise.interval.interval import Samples, Trace, create_monitor
from premise.interval.learningIMC import (
    build_learning_params_args_parser,
    initial_interval_learning,
    interval_learning,
    premilinaries,
)
from premise.interval.loading import build_suo, build_suo_args_parser
from premise.interval.loss import Distance, distance_measures
from premise.system import SystemUnderObservation


class RefinementStoppingCondition(ABC):
    def check(self, interval, initial_interval) -> None | tuple[Samples, Samples]:
        raise NotImplementedError(
            "RefinementStoppingCondition is an abstract class, please implement the check method"
        )

    def stats(self) -> dict[str, Any]:
        raise NotImplementedError(
            "RefinementStoppingCondition is an abstract class, please implement the stats method"
        )


class SampleCountStoppingCondition(RefinementStoppingCondition):
    def __init__(self, suo: SystemUnderObservation, sample_count: int):
        self.suo = suo
        self.sample_count = sample_count

    def check(self, interval, initial_interval) -> None | tuple[Samples, Samples]:
        if self.suo.stats()["sample_count"] >= self.sample_count:
            return None

        return [tuple()], []


class TargetDistanceStoppingCondition(RefinementStoppingCondition, ABC):
    def __init__(
        self,
        suo: SystemUnderObservation,
        length: int,
        horizon: int,
        amount: int,
        distance: Distance,
        verbose: int = 0,
    ):
        self.suo = suo
        self.length = length
        self.horizon = horizon
        self.amount = amount
        self.distance_func = distance
        self.verbose = verbose

        self.distances = []

        self.target_monitor = suo.create_target_monitor()

    def distance(
        self, interval, initial_interval
    ) -> tuple[float, list[tuple[Trace, tuple[float, float]]], Samples]:
        # Generate samples with weights
        samples_with_prob = self.suo.generate_random_traces_with_prob(
            [], self.length, self.amount
        )
        total_prob = sum(p for _, p in samples_with_prob)
        weights = {s: float(p / total_prob) for s, p in samples_with_prob}
        samples = [s[0] for s in samples_with_prob]

        # Build the premise monitor on the learned model
        mon, observation_map, _, _ = create_monitor(
            interval,
            initial_interval,
            "min",
            True,
            self.horizon,
        )

        # Run premise on the learned model
        monitored_risks = test_monitor(
            mon,
            samples,
            obs_func=lambda x: observation_map[x],
            skip_initial=True,
            with_tqdm=False,
        )

        # Run premise on the true model
        target_risks = test_monitor(
            self.target_monitor,
            samples,
            with_tqdm=False,
        )

        # Calculate the distance
        target_dist, target_all_dist = self.distance_func.distance(
            weights,
            {s: float(r) for s, r in target_risks.items()},
            {s: float(r) for s, r in monitored_risks.items()},
            all_distances=True,
        )

        self.distances.append(target_dist)

        return target_dist, target_all_dist, samples

    def stats(self) -> dict[str, Any]:
        return {
            "distances": self.distances,
        }


class ThresholdStoppingCondition(TargetDistanceStoppingCondition):
    def __init__(
        self,
        *args,
        threshold: float,
        patience: int,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.threshold = threshold
        self.patience = patience

        self.not_improved = 0

    def check(self, interval, initial_interval) -> None | tuple[Samples, Samples]:
        target_dist, target_all_dist, samples = self.distance(
            interval, initial_interval
        )

        if self.verbose > 0:
            print(f"Target distance: {target_dist} ? {self.threshold}")

        # If the distance is below the threshold, stop refinement
        if target_dist < self.threshold:
            self.not_improved += 1

            if self.verbose > 0:
                print(
                    f"Not improved for {self.not_improved} iterations, target distance: {target_dist:.4f} < {self.threshold:.4f}"
                )

            if self.not_improved >= self.patience:
                if self.verbose > 0:
                    print(
                        f"Stopping refinement at distance {target_dist} < {self.threshold}"
                    )
                return None

        # Otherwise, return the samples that are above the threshold
        interresting_traces = [t for t, (_, d) in target_all_dist if d > self.threshold]
        return ([t[:l] for t in interresting_traces for l in range(0, len(t))], samples)


class StabalizationStoppingCondition(TargetDistanceStoppingCondition):
    def __init__(
        self,
        *args,
        relative_deviation: float,
        patience: int,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.relative_deviation = relative_deviation
        self.patience = patience

        self.not_improved = 0

    def check(self, interval, initial_interval) -> None | tuple[Samples, Samples]:
        target_dist, target_all_dist, samples = self.distance(
            interval, initial_interval
        )

        mean_distance = np.mean(self.distances[-self.patience :])

        if self.verbose > 0:
            print(f"Target distance: {target_dist} ? {mean_distance}")

        rel_improvement = (mean_distance - target_dist) / mean_distance
        if rel_improvement < self.relative_deviation:
            self.not_improved += 1

            if self.verbose > 0:
                print(
                    f"Not improved for {self.not_improved} iterations, relative improvement: {rel_improvement:.4f}"
                )
        else:
            self.not_improved = 0

            if self.verbose > 0:
                print(f"Improved by {rel_improvement:.4f} in this iteration")

        if self.not_improved >= self.patience:
            if self.verbose > 0:
                print(
                    f"Stopping refinement at distance {target_dist} after not improving for {self.not_improved} iterations"
                )
            return None

        # Return samples with a distance above the full distance
        interresting_traces = [s for s, (_, d) in target_all_dist if d > target_dist]
        return [t[:l] for t in interresting_traces for l in range(0, len(t))], samples


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

    if verbose > 0:
        iteration = 0

    while True:
        if verbose > 0:
            print(
                f"-----------------------\nRefinement iteration {iteration} with {len(prefixes)} prefixes"
            )
            iteration += 1
        if verbose > 1:
            print(f"Prefixes: {prefixes}")

        amount = initial_learning_amount if prefixes == [tuple()] else learning_amount
        initial_samples = []
        samples = extra_samples
        for prefix in prefixes:
            s = suo.generate_random_traces(
                [s[1] for s in prefix],
                learning_length,
                amount,
            )
            if len(prefix) == 0:
                initial_samples += s

            samples += [t[len(prefix) :] for t in s]

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
            print(
                f"Finished learning with {len(samples)} additional samples (total: {suo.stats()['sample_count']})"
            )

        res = refinement_stopping_condition.check(interval, initial_interval)
        if res is not None:
            prefixes, extra_samples = res
        else:
            break

    return interval, initial_interval


def ref_main(args: argparse.Namespace):
    if args.conformence_length is None:
        args.conformence_length = args.sample_length

    suo = build_suo(args)

    distance = distance_measures[args.distance](args.distance_threshold)

    if args.stopping_criteria == "stabilization":
        ref_stop_cond = StabalizationStoppingCondition(
            suo,
            args.conformence_length,
            args.horizon,
            args.conformence_amount,
            distance,
            relative_deviation=args.stopping_deviation,
            patience=args.stopping_patience,
            verbose=args.verbose,
        )
    elif args.stopping_criteria == "threshold":
        ref_stop_cond = ThresholdStoppingCondition(
            suo,
            args.conformence_length,
            args.horizon,
            args.conformence_amount,
            distance,
            threshold=args.stopping_threshold,
            patience=args.stopping_patience,
            verbose=args.verbose,
        )
    elif args.stopping_criteria == "samples":
        ref_stop_cond = SampleCountStoppingCondition(
            suo,
            args.stopping_samples,
        )

    interval, initial_interval = refinement_learning(
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
        args.verbose,
    )

    if args.dump_stats:
        stats = ref_stop_cond.stats() | suo.stats()
        np.save(args.dump_stats, stats)  # type: ignore

    model_path = args.model_path
    if model_path is None:
        model_path = f"out/{suo.model_name}"
    numpy.save(f"{model_path}-initial_interval.npy", initial_interval)  # type: ignore
    numpy.save(f"{model_path}-interval.npy", interval)  # type: ignore

    return stats


def ref_args_parser():

    parser = argparse.ArgumentParser(description="Conformance checking")

    build_suo_args_parser(parser)

    learning_group = parser.add_argument_group("Learning")
    learning_group.add_argument(
        "-ll",
        "--sample_length",
        required=True,
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
        "-d",
        "--distance",
        choices=distance_measures.keys(),
        default="mae",
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
        "-ho", "--horizon", required=True, type=int, help="The horizon to monitor on"
    )
    conformence_group.add_argument(
        "-sc",
        "--stopping-criteria",
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

    parser.add_argument(
        "--dump-stats",
        type=str,
        help="Path to the file to dump stats to",
    )

    parser.add_argument("--verbose", "-v", action="count", default=0)

    build_learning_params_args_parser(parser)
    return parser


if __name__ == "__main__":
    parser = ref_args_parser()
    parsed_args = parser.parse_args()

    ref_main(parsed_args)
