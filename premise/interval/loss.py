from abc import ABC
from math import sqrt
from re import L
from typing import Any, Literal, overload

from premise.interval.interval import Trace


def uniform_weights(samples: list[list[tuple[int, int, bool]]]) -> dict[Any, float]:
    weights = {s: 1 / len(samples) for s in samples}
    return weights


class Distance(ABC):
    def __init__(self, *args, **kwargs):
        pass

    @overload
    def distance(
        self,
        weights: dict[Any, float],
        target_risks: dict[Any, float],
        risks: dict[Any, float],
        all_distances: Literal[False] = False,
    ) -> float: ...

    @overload
    def distance(
        self,
        weights: dict[Any, float],
        target_risks: dict[Any, float],
        risks: dict[Any, float],
        all_distances: Literal[True] = True,
    ) -> tuple[float, list[tuple[Trace, tuple[float, float]]]]: ...

    def distance(
        self,
        weights: dict[Any, float],
        target_risks: dict[Any, float],
        risks: dict[Any, float],
        all_distances=False,
    ) -> float | tuple[float, list[tuple[Trace, tuple[float, float]]]]:
        raise NotImplementedError(
            "The distance method must be implemented in subclasses."
        )


class MSEDistance(Distance):
    def distance(
        self,
        weights: dict[Any, float],
        target_risks: dict[Any, float],
        risks: dict[Any, float],
        all_distances=False,
    ) -> float | tuple[float, list[tuple[Trace, tuple[float, float]]]]:
        total = 0
        all_distances_list = []
        for key in target_risks.keys():
            if key in risks:
                total += weights[key] * (target_risks[key] - risks[key]) ** 2
                all_distances_list.append(
                    (key, (weights[key], (target_risks[key] - risks[key]) ** 2))
                )
            else:
                raise ValueError(
                    f"Key {key} not found in risks. Please check the input data."
                )

        if all_distances:
            return total, all_distances_list
        return total


class RMSEDistance(Distance):
    def distance(
        self,
        weights: dict[Any, float],
        target_risks: dict[Any, float],
        risks: dict[Any, float],
        all_distances=False,
    ) -> float | tuple[float, list[tuple[Trace, tuple[float, float]]]]:
        res = MSEDistance().distance(weights, target_risks, risks, all_distances)
        if isinstance(res, tuple):
            total, all_distances_list = res
            return sqrt(total), [(s, (w, sqrt(d))) for s, (w, d) in all_distances_list]
        return sqrt(res)


class MAEDistance(Distance):
    def distance(
        self,
        weights: dict[Any, float],
        target_risks: dict[Any, float],
        risks: dict[Any, float],
        all_distances=False,
    ) -> float | tuple[float, list[tuple[Trace, tuple[float, float]]]]:
        total = 0
        all_distances_list: list[tuple[Trace, tuple[float, float]]] = []
        for key in target_risks.keys():
            if key in risks:
                total += weights[key] * abs(target_risks[key] - risks[key])
                all_distances_list.append(
                    (key, (weights[key], abs(target_risks[key] - risks[key])))
                )
            else:
                raise ValueError(
                    f"Key {key} not found in risks. Please check the input data."
                )

        if all_distances:
            return total, all_distances_list
        return total


class ThresholdDistance(Distance):
    def __init__(self, threshold: float):
        self.threshold = threshold
        super().__init__()

    def distance(
        self,
        weights: dict[Any, float],
        target_risks: dict[Any, float],
        risks: dict[Any, float],
        all_distances=False,
    ):
        total = 0
        all_distances_list: list[tuple[Trace, tuple[float, float]]] = []
        for key in target_risks.keys():
            if key in risks:
                if risks[key] < self.threshold <= target_risks[key]:
                    total += weights[key] * (target_risks[key] - self.threshold)
                    all_distances_list.append(
                        (key, (weights[key], target_risks[key] - self.threshold))
                    )
                elif risks[key] > self.threshold >= target_risks[key]:
                    total += weights[key] * (self.threshold - target_risks[key])
                    all_distances_list.append(
                        (key, (weights[key], self.threshold - target_risks[key]))
                    )
                else:
                    all_distances_list.append((key, (weights[key], 0)))
            else:
                raise ValueError(
                    f"Key {key} not found in risks. Please check the input data."
                )

        if all_distances:
            return total, all_distances_list
        return total


distance_measures: dict[str, type[Distance]] = {
    "mse": MSEDistance,
    "rmse": RMSEDistance,
    "mae": MAEDistance,
    "thr": ThresholdDistance,
}
