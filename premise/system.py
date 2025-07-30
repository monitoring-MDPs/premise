from abc import ABC
from math import ceil
import pickle
from pathlib import Path
import random
from typing import Any, NoReturn, Optional

import stormpy
import stormpy.pomdp
from stormvogel import stormvogel_to_stormpy, Model, Path as SVPath, extensions

from premise.interval.interval import Samples, State, Trace
from premise.monitor import PremiseOptions, UnfoldingRiskAssessment, Monitor
from premise.trace_generator import ConditionalTraceGenerator
from premise.models import (
    ModelDescription,
    build_coarse_state_map,
    build_noaction_model_and_risk,
    build_state_and_transition_list,
    _analyse_model,
)
from premise.carla.model_info import get_states_and_transitions
from premise.sv_benchmarks import acas


class SystemUnderObservation(ABC):
    model_name: str

    def get_states_and_transitions(
        self, all_transitions: bool = False
    ) -> tuple[list[State], list[tuple[State, State]], list[State]]:
        raise NotImplementedError("This method should be overridden by subclasses")

    def generate_random_traces(
        self,
        observation_prefix: list[Any],
        length: int,
        amount: int = 1,
        initial_state: Optional[State] = None,
    ) -> Samples:
        raise NotImplementedError("This method should be overridden by subclasses")

    def generate_random_traces_with_prob(
        self,
        observation_prefix: list[Any],
        length: int,
        amount=1,
        initial_state: Optional[State] = None,
    ) -> list[tuple[Trace, Any]]:
        raise NotImplementedError("This method should be overridden by subclasses")

    def get_risk(self):
        raise NotImplementedError("This method should be overridden by subclasses")

    def create_target_monitor(self, dump_model=None) -> Monitor:
        raise NotImplementedError("This method should be overridden by subclasses")

    def trace_to_str(self, trace: Trace, gif_path=None) -> str:
        return " -> ".join([f"({s[0]}, {s[1]}, {s[2]})" for s in trace])

    def stats(self) -> dict[str, Any]:
        raise NotImplementedError("This method should be overridden by subclasses")


class MCSystemUnderObservation(SystemUnderObservation):
    def __init__(
        self,
        model_def: ModelDescription,
        name: str,
        options=PremiseOptions(),
        model=None,
        risks=None,
    ):
        self._model_def = model_def

        if model is not None and risks is not None:
            self._model = model
        else:
            self._model, self.risk = build_noaction_model_and_risk(
                model_def, options, risks is not None, pre_build_model=model
            )

        if risks is not None:
            self.risk = risks

        self.model_name = name
        self._ctr = ConditionalTraceGenerator(
            self._model, target_label=model_def.target_label
        )

        self._sample_count = 0
        self._transition_count = 0

    def get_states_and_transitions(self, all_transitions: bool = False, **kwargs):
        return build_state_and_transition_list(
            self._model, self._model_def.target_label, all_transitions, **kwargs
        )

    def generate_random_traces(
        self,
        observation_prefix: list[Any],
        length: int,
        amount=1,
        initial_state: Optional[tuple[int, int, bool]] = None,
    ) -> Samples:
        self._sample_count += amount
        self._transition_count += length * amount
        samples = [
            self._ctr.generate_random_trace(
                observation_prefix,
                length,
                initial_state[0] if initial_state is not None else None,
            )
            for _ in range(amount)
        ]
        return [s[0] for s in samples]

    def generate_random_traces_with_prob(
        self,
        observation_prefix: list[Any],
        length: int,
        amount=1,
        initial_state: Optional[State] = None,
    ) -> list[tuple[Trace, Any]]:
        self._sample_count += amount
        self._transition_count += length * amount
        return [
            self._ctr.generate_random_trace(
                observation_prefix,
                length,
                initial_state[0] if initial_state is not None else None,
            )
            for _ in range(amount)
        ]

    def get_risk(self):
        return self.risk

    def create_target_monitor(self, dump_model=None) -> Monitor:
        expr_manager = stormpy.ExpressionManager()
        unfolder = stormpy.pomdp.create_observation_trace_unfolder(
            self._model, self.risk, expr_manager
        )
        ura = UnfoldingRiskAssessment(
            stormpy.Environment(), unfolder, dump_model_to=dump_model
        )
        mon = Monitor(ura, 1000000)
        return mon

    def trace_to_str(self, trace: Trace, gif_path=None) -> str:
        return "\n-> ".join(
            [
                f"{i}: {self._model.state_valuations.get_string(s).replace(' ', '')} "
                f"{{{self._model.observation_valuations.get_string(o).replace(' ', '')}}} ({b}) [{(s,o,b)}]"
                for i, (s, o, b) in enumerate(trace)
            ]
        )

    def stats(self) -> dict[str, Any]:
        return {
            "sample_count": self._sample_count,
            "transition_count": self._transition_count,
        }


class ACASSystemUnderObservation(MCSystemUnderObservation):
    def __init__(
        self, coarseness_factor: float, horizon: int, sim_coarse_factor: float = 10.0
    ):
        self._sv_model: Model = acas.build_acas_model(
            radius_coarse=ceil(acas.RADIUS_COARSE * coarseness_factor),
            radius_obs=ceil(acas.RADIUS_OBS * coarseness_factor),
            bearing_coarse=ceil(acas.BEARING_COARSE * coarseness_factor),
            bearing_obs=ceil(acas.BEARING_OBS * coarseness_factor),
            rel_heading_coarse=ceil(acas.REL_HEADING_COARSE * coarseness_factor),
            rel_heading_obs=ceil(acas.REL_HEADING_OBS * coarseness_factor),
            ego_speed_coarse=ceil(acas.EGO_SPEED_COARSE * coarseness_factor),
            ego_speed_obs=ceil(acas.EGO_SPEED_OBS * coarseness_factor),
            int_speed_coarse=ceil(acas.INT_SPEED_COARSE * coarseness_factor),
            int_speed_obs=ceil(acas.INT_SPEED_OBS * coarseness_factor),
        )
        self._model = stormvogel_to_stormpy(self._sv_model, exact=True)
        self._stormpy_to_storvogel_id = {
            v: k for k, v in self._sv_model.stormpy_id.items()
        }
        self._model_def = ModelDescription(Path(), "", "", "nmac")
        prop = stormpy.parse_properties(f'Pmax=? [F<={horizon} "nmac"]')
        self.risk = _analyse_model(self._model, prop[0]).get_values()
        self.model_name = f"ACAS_{coarseness_factor}"

        self._ctr = ConditionalTraceGenerator(self._model, target_label="nmac")

        self._sample_count = 0
        self._transition_count = 0

    def generate_random_traces(
        self,
        observation_prefix: Samples,
        length: int,
        amount=1,
        initial_state: State | None = None,
    ) -> Samples:
        samples = []
        initial_acas_state: Optional[acas.ACAState] = None
        if initial_state is not None:
            for sv_id, s_id in self._sv_model.stormpy_id.items():
                if s_id == initial_state[0]:
                    initial_acas_state = (
                        self._sv_model.states[sv_id].valuations["ACAState"].copy()
                    )
                    break
            else:
                raise ValueError(
                    f"Initial state {initial_state} not found in ACAS model states."
                )

        for _ in range(amount):
            acas_state = initial_acas_state or acas.ACAState(coarse=False)
            s = [self._ACAState_to_State(acas_state)]
            for _ in range(length):
                acas_state.step()
                s.append(self._ACAState_to_State(acas_state))

            samples.append(s)

        self._sample_count += amount
        self._transition_count += length * amount

        return samples

    def generate_random_traces_with_prob(
        self,
        observation_prefix: Samples,
        length: int,
        amount=1,
        initial_state: State | None = None,
    ) -> NoReturn:
        raise NotImplementedError(
            "ACASSystemUnderObservation does not support generating traces with probabilities."
        )

    def _ACAState_to_State(self, acas_state: acas.ACAState) -> State:
        sv_state = None
        for s in self._sv_model.states.value():
            if s.valuations["ACAState"] == acas_state:
                sv_state = s
                break
        else:
            raise ValueError(
                f"ACAState {acas_state} not found in stormvogel model states."
            )

        s_id = self._sv_model.stormpy_id[sv_state.id]

        return (s_id, self._model.get_observation(s_id), "nmac" in acas_state.labels())

    def trace_to_str(self, trace: Trace, gif_path=None) -> str:
        if gif_path is not None:
            path = SVPath(
                {
                    i: self._sv_model.states[self._stormpy_to_storvogel_id[s]]
                    for i, (s, _, _) in enumerate(trace)
                },
                self._sv_model,
            )
            filename = extensions.render_model_gif(
                self._sv_model,
                lambda s: s.valuations["ACAState"].draw(),
                filename=gif_path,
                path=path,
                fps=0.5,
            )
        return "\n-> ".join(
            [
                f"{i}: {self._sv_model.states[self._stormpy_to_storvogel_id[s]].valuations} "
                f"{{{self._sv_model.states[self._stormpy_to_storvogel_id[s]].observation}}} "
                f"({b}) [{(s,o,b)}]"
                for i, (s, o, b) in enumerate(trace)
            ]
        )


class CoarseMCSystemUnderObservation(MCSystemUnderObservation):
    def __init__(self, model_def: ModelDescription, name: str, sys_vars: list[str]):
        super().__init__(model_def, name)
        self.sys_vars = sys_vars
        self.coarse_state_map, self.state_coarse_map = build_coarse_state_map(
            self._model, self.sys_vars
        )

    def get_states_and_transitions(self, all_transitions: bool = False):
        return super().get_states_and_transitions(
            all_transitions, state_coarse_map=self.state_coarse_map
        )

    def generate_random_traces(
        self,
        observation_prefix: list[Any],
        length: int,
        amount=1,
        initial_state: Optional[State] = None,
    ) -> Samples:
        if initial_state is not None:
            initial_stormpy_states = self.coarse_state_map[initial_state[0]]
            initial_state = (
                random.choice(initial_stormpy_states),
                initial_state[1],
                initial_state[2],
            )

        fine_traces = super().generate_random_traces(
            observation_prefix, length, amount, initial_state
        )
        coarse_traces = [
            tuple((self.state_coarse_map[s], o, l) for s, o, l in t)
            for t in fine_traces
        ]
        return coarse_traces

    def generate_random_traces_with_prob(
        self,
        observation_prefix: list[Any],
        length: int,
        amount=1,
        initial_state: Optional[State] = None,
    ) -> list[tuple[Trace, Any]]:
        if initial_state is not None:
            initial_stormpy_states = self.coarse_state_map[initial_state[0]]
            initial_state = (
                random.choice(initial_stormpy_states),
                initial_state[1],
                initial_state[2],
            )

        fine_traces = super().generate_random_traces_with_prob(
            observation_prefix, length, amount, initial_state
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

    def trace_to_str(self, trace: Trace, gif_path=None) -> str:
        return "\n-> ".join(
            [
                f"{i}: {';'.join(map(str, s))} {{{self._model.observation_valuations.get_string(o).replace(' ', '')}}} ({b})"
                for i, (s, o, b) in enumerate(trace)
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
