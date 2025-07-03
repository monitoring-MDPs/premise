from copy import copy
from functools import cache
from typing import Optional, Self
from numpy.random import normal
from scipy.stats import norm


class CoarseValue:
    min_bound: float
    max_bound: float
    coarseness: int
    obs_coarseness: int
    loop: bool
    coarse_value: int
    coarse_obs: int
    value: float
    obs: float

    def __init__(
        self,
        min_bound: float,
        max_bound: float,
        coarseness: int,
        obs_coarseness: int,
        timestep: float,
        loop: bool = False,
        init_value: Optional[float] = None,
        coarse_value: Optional[int] = None,
    ):
        if coarse_value is not None and init_value is not None:
            raise ValueError("Cannot set both coarse_value and init_value")

        self.min_bound = min_bound
        self.max_bound = max_bound
        self.coarseness = coarseness
        self.obs_coarseness = obs_coarseness
        self.timestep = timestep
        self.loop = loop
        if init_value is not None:
            self.update(init_value)
        elif coarse_value is not None:
            self.update_coarse(coarse_value)
        else:
            self.coarse_value = 0
            self.coarse_obs = 0
            self.value = min_bound
            self.obs = min_bound

    def update(self, new_value: float):
        self.coarse_value = self._calc_coarse(new_value)
        self.value = self.calc_value(self.coarse_value)

        self.coarse_obs = self._calc_coarse(new_value, obs=True)
        self.obs = self.calc_value(self.coarse_obs, obs=True)

    def update_coarse(self, new_coarse_value: int):
        if not (0 <= new_coarse_value < self.coarseness):
            raise ValueError(
                f"Coarse value {new_coarse_value} out of bounds [0, {self.coarseness})"
            )
        self.coarse_value = new_coarse_value
        self.value = self.calc_value(new_coarse_value)

        self.coarse_obs = self._calc_coarse(self.value, obs=True)
        self.obs = self.calc_value(self.coarse_obs, obs=True)

    def _transform_to_coarse(self, value: float, obs: bool = False) -> float:
        return (
            (value - self.min_bound)
            / (self.max_bound - self.min_bound)
            * (self.obs_coarseness if obs else self.coarseness)
        )

    def _calc_coarse(self, value: float, obs: bool = False):
        if not self.loop:
            value = min(max(value, self.min_bound), self.max_bound)
        new_coarse = round(self._transform_to_coarse(value, obs=obs))
        if self.loop:
            new_coarse %= self.obs_coarseness if obs else self.coarseness

        return new_coarse

    def calc_value(self, coarse_value: int, obs: bool = False):
        return (
            coarse_value
            / (self.obs_coarseness if obs else self.coarseness)
            * (self.max_bound - self.min_bound)
            + self.min_bound
        )

    def copy_update(self, new_value):
        new = copy(self)
        new.update(new_value)
        return new

    def copy_coarse(self, coarse_value) -> Self:
        new = copy(self)
        new.update_coarse(coarse_value)
        return new

    def unif_distr(self) -> list[tuple[float, Self]]:
        return [
            (1 / self.coarseness, self.copy_coarse(i)) for i in range(self.coarseness)
        ]

    def at_max(self):
        return self.coarse_value == self.coarseness - 1

    def at_min(self):
        return self.coarse_value == 0

    def __hash__(self) -> int:
        return hash(self.coarse_value)

    def __str__(self) -> str:
        return f"{self.value:.2f} [{self.min_bound:.2f},{self.coarseness},{self.max_bound:.2f}]"

    def __repr__(self) -> str:
        return f"CoarseValue({self.min_bound}, {self.max_bound}, {self.coarseness}, {self.coarse_value})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, CoarseValue):
            return False
        return self.coarse_value == other.coarse_value


class CoarseGaussianValue(CoarseValue):
    mean: float
    std: float
    min_prob: float

    def __init__(
        self,
        min_bound: float,
        max_bound: float,
        coarseness: int,
        obs_coarseness: int,
        timestep: float,
        mean: float,
        std: float,
        min_prob: float,
        loop: bool = False,
        init_value: Optional[float] = None,
        coarse_value: Optional[int] = None,
    ):
        super().__init__(
            min_bound, max_bound, coarseness, obs_coarseness, timestep, loop, init_value
        )
        self.mean = mean
        self.std = std
        self.min_prob = min_prob
        if coarse_value is not None:
            self.coarse_value = coarse_value
            self.value = self.calc_value(coarse_value)
        if init_value is not None:
            self.coarse_value = self._calc_coarse(init_value)
            self.value = self.calc_value(self.coarse_value)

    def update(self, new_value):
        new_value = normal(new_value + self.mean, self.std)
        super().update(new_value)

    def distr(self, new_value):
        coarse_value = self._calc_coarse(new_value)
        distr = []
        for i in range(self.coarseness):
            prob = calc_prob(
                i,
                coarse_value
                + (self.mean / (self.max_bound - self.min_bound) * self.coarseness),
                self.std
                * self.timestep
                / (self.max_bound - self.min_bound)
                * self.coarseness,
            )
            if prob < self.min_prob:
                continue
            distr.append((prob, self.copy_coarse(i)))
        total_prob = sum([p for p, _ in distr])
        norm_distr = [(p / total_prob, v) for p, v in distr]
        return norm_distr

    def __str__(self):
        return f"{self.value:.2f} [{self.min_bound:.2f},{self.coarseness},{self.max_bound:.2f}]@[{self.mean:.2f},{self.std:.2f},{self.min_prob:.2f}]"

    def __repr__(self) -> str:
        return f"CoarseGaussianValue({self.coarse_value}, {self.min_bound}, {self.max_bound}, {self.coarseness}, {self.mean}, {self.std}, {self.min_prob})"


class NonCoarseValue(CoarseValue):
    def update(self, new_value: float):
        self.coarse_value = self._calc_coarse(new_value)
        self.value = new_value

        self.coarse_obs = self._calc_coarse(new_value, obs=True)
        self.obs = self.calc_value(self.coarse_obs, obs=True)


class NonCoarseGaussianValue(CoarseGaussianValue):
    def update(self, new_value: float):
        new_value = normal(new_value + self.mean, self.std)
        self.coarse_value = self._calc_coarse(new_value)
        self.value = new_value

        self.coarse_obs = self._calc_coarse(new_value, obs=True)
        self.obs = self.calc_value(self.coarse_obs, obs=True)


@cache
def calc_prob(val: int, mean, std):
    return norm.cdf(val + 0.5, mean, std) - norm.cdf(val - 0.5, mean, std)
