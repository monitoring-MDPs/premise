import logging
import os
from pathlib import Path
import shutil

import stormpy as sp
import stormpy.simulator
from tqdm import tqdm

import models

logger = logging.getLogger(__name__)


def make_simulation_wrapper(model, length=None, seed=None):
    logger.info("Initialize simulator...")
    simulator = sp.simulator.create_simulator(model, seed)
    if length is None:
        return SimulationTraceGenerator(simulator)
    else:
        return FixedLengthSimulationTraceGenerator(simulator, length)


class SimulationTraceGenerator:
    def __init__(self, simulator):
        self._simulator = simulator

    def initialize(self) -> int:
        observation, _, _ = self._simulator.restart()
        return observation

    def step(self) -> int:
        observation, _, _ = self._simulator.random_step()
        return observation

    def set_seed(self, new_seed: int) -> None:
        self._simulator.set_seed(new_seed)

    def generate_random_trace(self, length: int) -> list[int]:
        trace = [self.initialize()]
        for i in range(length):
            trace.append(self.step())
        return trace


class FixedLengthSimulationTraceGenerator(SimulationTraceGenerator):
    def __init__(self, simulator, length):
        super().__init__(simulator)
        self._length = length
        self._steps_since_restart = 0

    def initialize(self):
        self._steps_since_restart = 0
        return super().initialize()

    def step(self):
        self._steps_since_restart += 1
        return super().step()

    def finished(self):
        return self._steps_since_restart >= self._length

    @property
    def max_length(self):
        return self._length


class FileCachedSimulationTraceGenerator(FixedLengthSimulationTraceGenerator):
    def __init__(self, stg: FixedLengthSimulationTraceGenerator, cache_file: str):
        super().__init__(stg._simulator, stg._length)
        self._stg = stg
        self._cache_file = cache_file
        self._cached_trace = self._load_trace()
        self._new_trace = []

    def _load_trace(self):
        if Path(self._cache_file).exists():
            with open(self._cache_file, "r") as f:
                trace = [int(line.strip()) for line in f.readlines()]
            return trace
        return None

    def _save_trace(self, trace):
        with open(self._cache_file, "w") as f:
            for observation in trace:
                f.write(f"{observation}\n")

    def initialize(self) -> int:
        if self._cached_trace is not None:
            self._steps_since_restart = 0
            return self._cached_trace[self._steps_since_restart]
        else:
            obs = super().initialize()
            self._new_trace.append(obs)
            return obs

    def step(self) -> int:
        if self._cached_trace is not None:
            self._steps_since_restart += 1
            return self._cached_trace[self._steps_since_restart]
        else:
            observation = super().step()
            self._new_trace.append(observation)
            if self.finished():
                self._save_trace(self._new_trace)
            return observation


def pre_generate_traces(
    name,
    path,
    constants,
    risk_property,
    options,
    trace_length: int,
    base_cache_path: Path,
    seed_list: list[int],
):
    print(f"Pre-generating traces for model {path}...")

    cached_seeds = []
    cache_path = base_cache_path / f"simulator-caches-{name}"
    os.makedirs(cache_path, exist_ok=True)

    timestamps = sorted(
        [
            folder.name.replace("exp-", "")
            for folder in base_cache_path.parent.glob("exp-*")
        ]
    )

    if len(timestamps) >= 2:
        newest_timestamp = timestamps[-2]
        #  The newest experiment is the current one
        previous_cache_path = (
            base_cache_path.parent
            / f"exp-{newest_timestamp}"
            / f"simulator-caches-{name}"
        )

        if previous_cache_path.exists():
            # copy existing caches
            for seed in seed_list:
                previous_cache_file = previous_cache_path / f"simulator-cache-{seed}"
                new_cache_file = cache_path / f"simulator-cache-{seed}"
                if previous_cache_file.exists() and not new_cache_file.exists():
                    shutil.copy(previous_cache_file, new_cache_file)
                    cached_seeds.append(seed)

        if len(cached_seeds) == len(seed_list):
            print("All traces already cached.")
            return

    model, _ = models.build_model_and_risk(
        models.ModelDescription(path, constants, risk_property), options
    )

    for seed in seed_list:
        if seed in cached_seeds:
            continue

        sim = sp.simulator.create_simulator(model, seed)
        stg = FixedLengthSimulationTraceGenerator(sim, trace_length)
        cache_file = cache_path / f"simulator-cache-{seed}"
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        cached_stg = FileCachedSimulationTraceGenerator(stg, str(cache_file))

        cached_stg.initialize()
        while not cached_stg.finished():
            cached_stg.step()
