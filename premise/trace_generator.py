import logging

import stormpy as sp
from logging import getLogger

logger = logging.getLogger(__name__)

def make_simulation_wrapper(model, risk_structure):
    logger.info("Initialize simulator...")
    simulator = sp.simulator.create_simulator(model)
    return SimulationTraceGenerator(simulator)


class SimulationTraceGenerator:
    def __init__(self, simulator):
        self._simulator = simulator

    def initialize(self):
        observation, _, _ = self._simulator.restart()
        return observation

    def step(self):
        observation, _, _ = self._simulator.random_step()
        return observation

    def generate_random_trace(self, length) -> list[int]:
        trace = []
        trace.append(self.initialize())
        for i in range(length):
            trace.append(self.step())
        return trace
