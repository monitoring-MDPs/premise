import logging
import random
import trace

import stormpy.simulator
from stormpy import Rational

logger = logging.getLogger(__name__)


def make_simulation_wrapper(model, length=None):
    logger.info("Initialize simulator...")
    simulator = stormpy.simulator.create_simulator(model)
    if length is None:
        return SimulationTraceGenerator(simulator)
    else:
        return FixedLengthSimulationTraceGenerator(simulator, length)


class ConditionalTraceGenerator:
    def __init__(self, model, target_label: str) -> None:
        self.model = model
        self.current_state = model.initial_states[0]
        self.target_label = target_label

    def set_seed(self, new_seed: int) -> None:
        random.seed(new_seed)

    def initialize(self) -> int:
        self.current_state = self.model.initial_states[0]
        return self.model.get_observation(self.current_state)

    def step(self, action=None) -> int:
        state = self.model.states[self.current_state]
        if action is None:
            action = random.choice(list(state.actions)).id

        probability = Rational(random.random())
        total_prob = Rational(0.0)
        for transition in state.actions[action].transitions:
            total_prob += transition.value()
            if total_prob > probability:
                self.current_state = transition.column
                break

        return self.model.get_observation(self.current_state)

    def conditional_step(self, observation, ignore_states=None, action=None):
        if ignore_states is None:
            ignore_states = []

        state = self.model.states[self.current_state]

        # Build a dictionary of unnormalized probabilities for each possible next state with the given observation
        conditional_state_probs = {}
        for action in state.actions:
            for transition in action.transitions:
                if (
                    transition.column not in ignore_states
                    and self.model.get_observation(transition.column) == observation
                ):
                    if transition.column in conditional_state_probs:
                        conditional_state_probs[transition.column] += transition.value()
                    else:
                        conditional_state_probs[transition.column] = transition.value()

        if len(conditional_state_probs) == 0:
            raise ValueError(
                f"No transition found for observation {observation} in state {self.current_state}"
            )

        probability = Rational(random.random()) * sum(conditional_state_probs.values())
        total_prob = Rational(0.0)
        for state_id, prob in conditional_state_probs.items():
            total_prob += prob
            if total_prob > probability:
                self.current_state = state_id
                break

        return self.model.get_observation(self.current_state)

    def generate_random_trace(
        self, observation_prefix: list[int], length: int
    ) -> list[tuple[int, int, bool]]:
        init_obs = self.initialize()
        init_state = self.current_state
        has_label = self.model.labeling.has_state_label(self.target_label, init_state)
        if len(observation_prefix) > 0 and init_obs != observation_prefix[0]:
            raise ValueError("Initial observation does not match prefix")
        res = self._generate_random_trace_rec(observation_prefix[1:], length - 1)
        if res is None:
            raise ValueError("Could not generate trace with given prefix")

        return [(init_state, init_obs, has_label)] + res

    def _generate_random_trace_rec(
        self, observation_prefix: list[int], length: int
    ) -> list[tuple[int, int, bool]] | None:
        if length == 0:
            return []
        elif len(observation_prefix) == 0:
            step_obs = self.step()
            new_state = self.current_state
            res = self._generate_random_trace_rec([], length - 1)
            if res is None:
                return None
            return [
                (
                    new_state,
                    step_obs,
                    self.model.labeling.has_state_label(self.target_label, new_state),
                )
            ] + res
        else:
            old_state = self.current_state
            bad_states = []
            while True:
                try:
                    cond_step_obs = self.conditional_step(
                        observation_prefix[0], ignore_states=bad_states
                    )
                    new_state = self.current_state
                    res = self._generate_random_trace_rec(
                        observation_prefix[1:], length - 1
                    )
                    if res is not None:
                        return [
                            (
                                new_state,
                                cond_step_obs,
                                self.model.labeling.has_state_label(
                                    self.target_label, old_state
                                ),
                            )
                        ] + res
                    else:
                        bad_states.append(self.current_state)
                        self.current_state = old_state
                except ValueError:
                    break
            self.current_state = old_state
            return None


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
