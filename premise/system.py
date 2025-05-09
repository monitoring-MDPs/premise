from abc import ABC
import pickle
from typing import Any

import stormpy
import stormpy.pomdp

from premise.interval.interval import Samples, State, Trace
from premise.monitor import PremiseOptions, UnfoldingRiskAssessment, Monitor
from premise.trace_generator import ConditionalTraceGenerator
from premise.models import (
    ModelDescription,
    build_coarse_state_map,
    build_noaction_model_and_risk,
    build_state_and_transition_list,
)
from premise.carla.model_info import get_states_and_transitions


class SystemUnderObservation(ABC):
    model_name: str

    def get_states_and_transitions(
        self, all_transitions: bool = True
    ) -> tuple[list[State], list[tuple[State, State]], list[State]]:
        raise NotImplementedError("This method should be overridden by subclasses")

    def generate_random_traces(
        self,
        observation_prefix: list[Any],
        length: int,
        amount: int = 1,
    ) -> Samples:
        raise NotImplementedError("This method should be overridden by subclasses")

    def generate_random_traces_with_prob(
        self,
        observation_prefix: list[Any],
        length: int,
        amount=1,
    ) -> list[tuple[Trace, Any]]:
        raise NotImplementedError("This method should be overridden by subclasses")

    def get_risk(self):
        raise NotImplementedError("This method should be overridden by subclasses")

    def create_target_monitor(self, dump_model=None) -> Monitor:
        raise NotImplementedError("This method should be overridden by subclasses")

    def trace_to_str(self, trace: Trace) -> str:
        return " -> ".join([f"({s[0]}, {s[1]}, {s[2]})" for s in trace])

    def stats(self) -> dict[str, Any]:
        raise NotImplementedError("This method should be overridden by subclasses")


class MCSystemUnderObservation(SystemUnderObservation):
    def __init__(self, model_def: ModelDescription, name: str):
        self._model_def = model_def
        self._model, self.risk = build_noaction_model_and_risk(
            model_def,
            PremiseOptions(),
        )
        self.model_name = name
        self._ctr = ConditionalTraceGenerator(
            self._model, target_label=model_def.target_label
        )

        self._sample_count = 0

    def get_states_and_transitions(self, all_transitions: bool = True, **kwargs):
        return build_state_and_transition_list(
            self._model, self._model_def.target_label, all_transitions, **kwargs
        )

    def generate_random_traces(
        self, observation_prefix: list[Any], length: int, amount=1
    ) -> Samples:
        self._sample_count += amount
        samples = [
            self._ctr.generate_random_trace(observation_prefix, length)
            for _ in range(amount)
        ]
        return [s[0] for s in samples]

    def generate_random_traces_with_prob(
        self, observation_prefix: list[Any], length: int, amount=1
    ) -> list[tuple[Trace, Any]]:
        self._sample_count += amount
        return [
            self._ctr.generate_random_trace(observation_prefix, length)
            for _ in range(amount)
        ]

    def get_risk(self):
        return self.risk

    def create_target_monitor(self, dump_model=None) -> Monitor:
        expr_manager = stormpy.ExpressionManager()  # type: ignore
        unfolder = stormpy.pomdp.create_observation_trace_unfolder(
            self._model, self.risk, expr_manager
        )
        ura = UnfoldingRiskAssessment(stormpy.Environment(), unfolder, dump_model_to=dump_model)  # type: ignore
        mon = Monitor(ura, 1000000)
        return mon

    def trace_to_str(self, trace: Trace) -> str:
        return "\n-> ".join(
            [
                f"{self._model.state_valuations.get_string(s).replace(' ', '')} {{{self._model.observation_valuations.get_string(o).replace(' ', '')}}} ({b})"
                for (s, o, b) in trace
            ]
        )

    def stats(self) -> dict[str, Any]:
        return {
            "sample_count": self._sample_count,
        }


class CoarseMCSystemUnderObservation(MCSystemUnderObservation):
    def __init__(self, model_def: ModelDescription, name: str, sys_vars: list[str]):
        super().__init__(model_def, name)
        self.sys_vars = sys_vars
        self.coarse_state_map, self.state_coarse_map = build_coarse_state_map(
            self._model, self.sys_vars
        )

    def get_states_and_transitions(self, all_transitions: bool = True):
        return super().get_states_and_transitions(
            all_transitions, state_coarse_map=self.state_coarse_map
        )

    def generate_random_traces(
        self, observation_prefix: list[Any], length: int, amount=1
    ) -> Samples:
        fine_traces = super().generate_random_traces(observation_prefix, length, amount)
        coarse_traces = [
            tuple((self.state_coarse_map[s], o, l) for s, o, l in t)
            for t in fine_traces
        ]
        return coarse_traces

    def generate_random_traces_with_prob(
        self, observation_prefix: list[Any], length: int, amount=1
    ) -> list[tuple[Trace, Any]]:
        fine_traces = super().generate_random_traces_with_prob(
            observation_prefix, length, amount
        )
        coarse_traces = [
            (tuple((self.state_coarse_map[s], o, l) for s, o, l in t), p)
            for t, p in fine_traces
        ]
        return coarse_traces

    def get_risk(self):
        raise NotImplementedError(
            "CoarseMCSystemUnderObservation does not support risk assessment on its states"
        )

    def trace_to_str(self, trace: Trace) -> str:
        return "\n-> ".join(
            [
                f"{';'.join(s)} {{{self._model.observation_valuations.get_string(o).replace(' ', '')}}} ({b})"
                for (s, o, b) in trace
            ]
        )


class CarlaPreSampledSystemUnderObservation(SystemUnderObservation):
    def __init__(
        self, sample_paths: list[str], condition_sample_paths: dict[str, str] = {}
    ):
        self.model_name = "CarlaPS"
        self.sample_paths = sample_paths
        self.condition_sample_paths = condition_sample_paths

        self.samples = []
        for s_p in sample_paths:
            with open(s_p, "rb") as f:
                self.samples += pickle.load(f)

        self.sample_index = 0

    def get_states_and_transitions(
        self, all_transitions: bool = True
    ) -> tuple[list[State], list[tuple[State, State]], list[State]]:
        return get_states_and_transitions()

    def generate_random_traces(
        self,
        observation_prefix: list[Any],
        length: int,
        amount: int = 1,
    ) -> Samples:
        if observation_prefix != []:
            raise ValueError("No prefix support yet")

        if self.sample_index + amount > len(self.samples):
            samples = (
                self.samples[min(len(self.samples), self.sample_index + amount) :]
                + self.samples[: self.sample_index + amount % len(self.samples)]
            )
            self.sample_index = (self.sample_index + amount) % len(self.samples)
            return [tuple(s[:length]) for s in samples]
        else:
            samples = self.samples[self.sample_index : self.sample_index + amount]
            self.sample_index += amount
            return [tuple(s[:length]) for s in samples]

        # return sample(self.scenic_path,amount,length)

    def generate_random_traces_with_prob(
        self,
        observation_prefix: list[Any],
        length: int,
        amount=1,
    ) -> list[tuple[Trace, Any]]:
        return [
            (t, 1 / amount)
            for t in self.generate_random_traces(observation_prefix, length, amount)
        ]

    def create_target_monitor(self, dump_model=None) -> Monitor:
        raise NotImplementedError("This method should be overridden by subclasses")

    def stats(self) -> dict[str, Any]:
        raise NotImplementedError("This method should be overridden by subclasses")


class CarlaSimSystemUnderObservation(SystemUnderObservation):
    def __init__(self, scenic_path) -> None:
        super().__init__()
        self.model_name = "CarlaSim"
        self.scenic_path = scenic_path

    def get_states_and_transitions(
        self, all_transitions: bool = True
    ) -> tuple[list[State], list[tuple[State, State]], list[State]]:
        return get_states_and_transitions()

    def generate_random_traces(
        self,
        observation_prefix: list[Any],
        length: int,
        amount: int = 1,
    ) -> Samples:
        from premise.carla.sample import sample

        if observation_prefix != []:
            raise ValueError("No prefix support yet")

        return sample(self.scenic_path, amount, length)

    def generate_random_traces_with_prob(
        self,
        observation_prefix: list[Any],
        length: int,
        amount=1,
    ) -> list[tuple[Trace, Any]]:
        return [
            (t, 1 / amount)
            for t in self.generate_random_traces(observation_prefix, length, amount)
        ]

    def create_target_monitor(self, dump_model=None) -> Monitor:
        raise NotImplementedError("This method should be overridden by subclasses")

    def stats(self) -> dict[str, Any]:
        raise NotImplementedError("This method should be overridden by subclasses")
