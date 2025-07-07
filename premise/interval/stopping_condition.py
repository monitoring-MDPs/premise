from abc import ABC
from collections import defaultdict
import logging
from typing import Any, Optional

import numpy as np

from premise.interval.conformence import test_monitor
from premise.interval.loss import Distance
from premise.interval.interval import MonitorComponents, Samples, Trace, create_monitor
from premise.system import SystemUnderObservation
from premise.interval.utils import logger


class DistanceCalculator(ABC):
    trace_state_risk_widths: dict[Trace, float] = defaultdict(float)
    mon_comps: Optional[MonitorComponents] = None

    def distance(
        self, interval, initial_interval, samples_with_prob
    ) -> tuple[float, list[tuple[Trace, tuple[float, float]]], Samples]:
        raise NotImplementedError(
            "DistanceCalculator is an abstract class, please implement the distance method"
        )


class TargetDistanceCalculator(DistanceCalculator):
    def __init__(
        self,
        suo: SystemUnderObservation,
        horizon: int,
        distance: Distance,
        use_exact: bool = False,
        precision: float = 1e-6,
        verbose: int = 0,
    ):
        self.suo = suo
        self.horizon = horizon
        self.distance_func = distance
        self.use_exact = use_exact
        self.precision = precision
        self.verbose = verbose

        self.traces = []

        self.target_monitor = suo.create_target_monitor()

        self.previous_risks = {}
        self.previous_weights = {}
        self.trace_risk_widths = defaultdict(float)

    def distance(
        self, interval, initial_interval, samples_with_prob
    ) -> tuple[float, list[tuple[Trace, tuple[float, float]]], Samples]:
        total_prob = sum(p for _, p in samples_with_prob)
        weights = {s: float(p / total_prob) for s, p in samples_with_prob}
        samples = [s[0] for s in samples_with_prob]

        # Build the premise monitor on the learned model
        mon, mon_comps = create_monitor(
            interval,
            initial_interval,
            "min",
            True,
            self.horizon,
            use_exact=self.use_exact,
            precision=self.precision,
        )
        self.mon_comps = mon_comps

        if self.verbose > 0:
            logger.info(f"Created all monitors, now testing them")

        # Run premise on the learned model
        monitored_risks = test_monitor(
            mon,
            samples,
            obs_func=lambda x: mon_comps.observation_map[x],
            skip_initial=True,
            with_tqdm=False,
        )

        # Run premise on the true model
        target_risks = test_monitor(
            self.target_monitor,
            samples,
            with_tqdm=False,
        )

        self.previous_risks = target_risks
        self.previous_weights = weights

        # Calculate the distance
        target_dist, target_all_dist = self.distance_func.distance(
            weights,
            {s: float(r) for s, r in target_risks.items()},
            {s: float(r) for s, r in monitored_risks.items()},
            all_distances=True,
        )

        target_all_dist_w_risk = [
            (t, (p, d, float(monitored_risks[t]), float(target_risks[t])))  # type: ignore
            for t, (p, d) in target_all_dist
        ]
        self.traces.append(target_all_dist_w_risk)

        return target_dist, target_all_dist, samples


class IntervalWidthCalculator(DistanceCalculator):
    def __init__(
        self,
        suo: SystemUnderObservation,
        horizon: int,
        distance: Distance,
        use_exact: bool = False,
        precision: float = 1e-6,
        verbose: int = 0,
    ):
        self.suo = suo
        self.horizon = horizon
        self.distance_func = distance
        self.use_exact = use_exact
        self.precision = precision
        self.verbose = verbose

        self.traces = []

        self.previous_risks = {}
        self.previous_weights = {}
        self.trace_state_risk_widths = {}

    def distance(
        self, interval, initial_interval, samples_with_prob
    ) -> tuple[float, list[tuple[Trace, tuple[float, float]]], Samples]:
        # Generate samples with weights
        total_prob = sum(p for _, p in samples_with_prob)
        weights = {s: float(p / total_prob) for s, p in samples_with_prob}
        samples = [s[0] for s in samples_with_prob]

        logger.info(f"Staring building monitors")

        # Build the premise monitor on the learned model
        max_mon, max_mon_comps = create_monitor(
            interval,
            initial_interval,
            "min",
            True,
            self.horizon,
            use_exact=self.use_exact,
            precision=self.precision,
        )

        logger.info(f"Created max monitor")

        min_mon, min_mon_comps = create_monitor(
            interval,
            initial_interval,
            "max",
            True,
            self.horizon,
            use_exact=self.use_exact,
            precision=self.precision,
        )

        self.mon_comps = max_mon_comps

        logger.info(f"Created all monitors, now testing them")

        # Run premise on the learned model
        min_monitored_risks = test_monitor(
            min_mon,
            samples,
            obs_func=lambda x: min_mon_comps.observation_map[x],
            skip_initial=True,
            with_tqdm=False,
        )

        logger.info(f"Monitored risks for min monitor: max={max(min_monitored_risks)}")

        max_monitored_risks = test_monitor(
            max_mon,
            samples,
            obs_func=lambda x: max_mon_comps.observation_map[x],
            skip_initial=True,
            with_tqdm=False,
        )

        logger.info(f"Monitored risks for max monitor: min={min(max_monitored_risks)}")

        # Calculate the distance
        target_dist, target_all_dist = self.distance_func.distance(
            weights,
            {s: float(r) for s, r in max_monitored_risks.items()},
            {s: float(r) for s, r in min_monitored_risks.items()},
            all_distances=True,
        )

        target_all_dist_w_risk = [
            (t, (p, d, float(min_monitored_risks[t]), float(max_monitored_risks[t])))
            for t, (p, d) in target_all_dist
        ]
        self.traces.append(target_all_dist_w_risk)

        # Risks are stored as point intervals, thus .upper() gives the risk of the state
        self.trace_state_risk_widths = {
            t: float(max_mon_comps.risks[max_mon_comps.state_index_map[t[-1]]].upper())
            - float(min_mon_comps.risks[min_mon_comps.state_index_map[t[-1]]].upper())
            for t in samples
        }

        return target_dist, target_all_dist, samples


class RefinementStoppingCondition(ABC):
    def __init__(
        self,
        suo: SystemUnderObservation,
        distance_calculator: DistanceCalculator,
        prefix_amount: int = 10,
        verbose: int = 0,
        length: int = 0,
        amount: int = 0,
    ):
        self.suo = suo
        self.distance_calculator = distance_calculator
        self.prefix_amount = prefix_amount
        self.distances = []
        self.traces = []
        self.verbose = verbose
        self.length = length
        self.amount = amount

    def check(self, interval, initial_interval) -> None | tuple[Samples, Samples]:
        raise NotImplementedError(
            "RefinementStoppingCondition is an abstract class, please implement the check method"
        )

    def _generate_traces(self):
        samples_with_prob = self.suo.generate_random_traces_with_prob(
            [], self.length, self.amount
        )
        return samples_with_prob

    def _generate_prefixes(
        self, interesting_traces: list[Trace], splits: Optional[list[float]] = None
    ):
        prefixes = []
        if splits is None:
            for t in interesting_traces:
                for l in np.linspace(
                    0.0, float(len(prefixes)), self.prefix_amount, endpoint=True
                ):
                    prefixes.append(t[: round(l)])
        else:
            for t, split in zip(interesting_traces, splits):
                prefixes += [()] * round(split * self.prefix_amount) + [t] * round(
                    (1 - split) * self.prefix_amount
                )
        return prefixes

    def stats(self) -> dict[str, Any]:
        return {
            "distances": self.distances,
            "dist_traces": self.traces,
        }


class SampleCountStoppingCondition(RefinementStoppingCondition):
    def __init__(
        self,
        suo: SystemUnderObservation,
        distance_calculator: DistanceCalculator,
        transition_count: int,
        refine_amount: int,
        verbose: int,
        length: int,
        amount: int,
        learning_length: int,
        prefix_amount: int = 10,
    ):
        super().__init__(
            suo, distance_calculator, prefix_amount, verbose, length, amount
        )
        self.transition_count = transition_count
        self.refine_amount = refine_amount
        self.learning_length = learning_length

    def check(self, interval, initial_interval) -> None | tuple[Samples, Samples]:
        pre_sampling_transition_count = self.suo.stats()["transition_count"]
        samples_with_prob = self._generate_traces()
        dist, dist_traces, samples = self.distance_calculator.distance(
            interval, initial_interval, samples_with_prob
        )
        self.traces.append(dist_traces)
        self.distances.append(dist)

        if pre_sampling_transition_count >= self.transition_count:
            if self.verbose > 0:
                logger.info(
                    f"Pre-sampling transition count {pre_sampling_transition_count} >= target {self.transition_count}, stopping refinement."
                )
            return None

        additional_samples = (
            self.transition_count / self.learning_length / self.prefix_amount
            - len(samples)
        )

        if (
            self.transition_count - self.suo.stats()["transition_count"]
            < additional_samples * self.learning_length
        ):
            additional_samples = (
                self.transition_count - self.suo.stats()["transition_count"]
            ) / self.learning_length

        return [tuple()] * int(
            np.ceil(additional_samples / self.refine_amount)
        ), samples


class ThresholdStoppingCondition(RefinementStoppingCondition):
    def __init__(
        self,
        suo: SystemUnderObservation,
        distance_calculator: DistanceCalculator,
        threshold: float,
        patience: int,
        prefix_amount: int = 10,
        verbose: int = 0,
        length: int = 0,
        amount: int = 0,
        use_splitting: bool = True,
    ):
        super().__init__(
            suo,
            distance_calculator,
            prefix_amount,
            verbose,
            length,
            amount,
        )
        self.threshold = threshold
        self.patience = patience

        self.not_improved = 0

        self.previous_interesting_traces = []
        self.use_splitting = use_splitting

    def check(self, interval, initial_interval) -> None | tuple[Samples, Samples]:
        samples_with_prob = self._generate_traces()
        target_dist, target_all_dist, samples = self.distance_calculator.distance(
            interval,
            initial_interval,
            self.previous_interesting_traces + samples_with_prob,
        )
        self.traces.append(target_all_dist)
        self.distances.append(target_dist)

        if self.verbose > 0:
            logger.info(f"Target distance: {target_dist} ? {self.threshold}")

        # If the distance is below the threshold, stop refinement
        if target_dist < self.threshold:
            self.not_improved += 1

            if self.verbose > 0:
                logger.info(
                    f"Not improved for {self.not_improved} iterations, target distance: {target_dist:.4f} < {self.threshold:.4f}"
                )

            if self.not_improved >= self.patience:
                if self.verbose > 0:
                    logger.info(
                        f"Stopping refinement at distance {target_dist} < {self.threshold}"
                    )
                return None
        else:
            self.not_improved = 0

        # Otherwise, return the samples that are above the threshold
        interesting_traces = [t for t, (_, d) in target_all_dist if d > self.threshold]
        splits = [
            (
                0.0
                if self.distance_calculator.trace_state_risk_widths[t] > self.threshold
                else 1.0
            )
            for t in interesting_traces
        ]
        self.previous_interesting_traces = [
            (t, w) for (t, w) in samples_with_prob if t in interesting_traces
        ]
        return (
            self._generate_prefixes(
                interesting_traces, splits if self.use_splitting else None
            ),
            samples,
        )


class StabilizationStoppingCondition(RefinementStoppingCondition):
    def __init__(
        self,
        suo: SystemUnderObservation,
        distance_calculator: DistanceCalculator,
        relative_deviation: float,
        patience: int,
        prefix_amount: int = 10,
        verbose: int = 0,
        length: int = 0,
        amount: int = 0,
    ):
        super().__init__(
            suo,
            distance_calculator,
            prefix_amount,
            verbose,
            length,
            amount,
        )
        self.relative_deviation = relative_deviation
        self.patience = patience

        self.not_improved = 0
        self.previous_interesting_traces = []

    def check(self, interval, initial_interval) -> None | tuple[Samples, Samples]:
        samples_with_prob = self._generate_traces()
        target_dist, target_all_dist, samples = self.distance_calculator.distance(
            interval,
            initial_interval,
            self.previous_interesting_traces + samples_with_prob,
        )
        self.traces.append(target_all_dist)
        self.distances.append(target_dist)

        mean_distance = np.mean(self.distances[-self.patience :])

        if self.verbose > 0:
            logger.info(f"Target distance: {target_dist} ? {mean_distance}")

        rel_improvement = (mean_distance - target_dist) / mean_distance
        if rel_improvement < self.relative_deviation:
            self.not_improved += 1

            if self.verbose > 0:
                logger.info(
                    f"Not improved for {self.not_improved} iterations, relative improvement: {rel_improvement:.4f}"
                )
        else:
            self.not_improved = 0

            if self.verbose > 0:
                logger.info(f"Improved by {rel_improvement:.4f} in this iteration")

        if self.not_improved >= self.patience:
            if self.verbose > 0:
                logger.info(
                    f"Stopping refinement at distance {target_dist} after not improving for {self.not_improved} iterations"
                )
            return None

        # Return samples with a distance above the full distance
        interesting_traces = [s for s, (_, d) in target_all_dist if d > target_dist]
        self.previous_interesting_traces = [
            (t, w) for (t, w) in samples_with_prob if t in interesting_traces
        ]
        return self._generate_prefixes(interesting_traces), samples
