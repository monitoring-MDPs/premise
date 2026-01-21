import logging

import stormpy as sp
import stormpy.simulator

logger = logging.getLogger(__name__)

def make_simulation_wrapper(model, length=None, seed=None):
    logger.info("Initialize simulator...")
    simulator = sp.simulator.create_simulator(model, seed=seed)
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

    def set_seed(self, new_seed : int) -> None:
        self._simulator.set_seed(new_seed)

    def generate_random_trace(self, length : int) -> list[int]:
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



