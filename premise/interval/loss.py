from abc import ABC
from math import sqrt
from re import L
from typing import Any, Literal, overload


def uniform_weights(samples: list[list[tuple[int, int, bool]]]) -> dict[Any, float]:
    weights = {s: 1 / len(samples) for s in samples}
    return weights


class Distance(ABC):
    @overload
    @staticmethod
    def distance(
        weights: dict[Any, float],
        target_risks: dict[Any, float],
        risks: dict[Any, float],
        all_distances: Literal[False] = False,
    ) -> float: ...

    @overload
    @staticmethod
    def distance(
        weights: dict[Any, float],
        target_risks: dict[Any, float],
        risks: dict[Any, float],
        all_distances: Literal[True] = True,
    ) -> tuple[float, list[tuple[Any, tuple[float, float]]]]: ...

    @staticmethod
    def distance(
        weights: dict[Any, float],
        target_risks: dict[Any, float],
        risks: dict[Any, float],
        all_distances=False,
    ) -> float | tuple[float, list[tuple[Any, tuple[float, float]]]]:
        raise NotImplementedError(
            "The distance method must be implemented in subclasses."
        )


class MSEDistance(Distance):
    @staticmethod
    def distance(
        weights: dict[Any, float],
        target_risks: dict[Any, float],
        risks: dict[Any, float],
        all_distances=False,
    ) -> float | tuple[float, list[tuple[Any, tuple[float, float]]]]:
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
    @staticmethod
    def distance(
        weights: dict[Any, float],
        target_risks: dict[Any, float],
        risks: dict[Any, float],
        all_distances=False,
    ) -> float | tuple[float, list[tuple[Any, tuple[float, float]]]]:
        res = MSEDistance.distance(weights, target_risks, risks, all_distances)
        if isinstance(res, tuple):
            total, all_distances_list = res
            return sqrt(total), [(s, (w, sqrt(d))) for s, (w, d) in all_distances_list]
        return sqrt(res)


class MAEDistance(Distance):
    @staticmethod
    def distance(
        weights: dict[Any, float],
        target_risks: dict[Any, float],
        risks: dict[Any, float],
        all_distances=False,
    ) -> float | tuple[float, list[tuple[Any, tuple[float, float]]]]:
        total = 0
        all_distances_list: list[tuple[Any, tuple[float, float]]] = []
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


distance_measures: dict[str, type[Distance]] = {
    "mse": MSEDistance,
    "rmse": RMSEDistance,
    "mae": MAEDistance,
}
