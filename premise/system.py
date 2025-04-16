from abc import ABC

import stormpy
import stormpy.pomdp

from premise.monitor import PremiseOptions, UnfoldingRiskAssessment, Monitor
from premise.trace_generator import ConditionalTraceGenerator
from premise.models import (
    ModelDescription,
    build_model_and_risk,
    build_state_and_transition_list,
)


class SystemUnderObservation(ABC):
    model_name: str

    def get_states_and_transitions(self) -> tuple[list, list[tuple]]:
        raise NotImplementedError("This method should be overridden by subclasses")

    def generate_random_traces(
        self, observation_prefix: list, length: int, amount: int = 1
    ) -> list[list[tuple[int, int, bool]]]:
        raise NotImplementedError("This method should be overridden by subclasses")

    def get_risk(self):
        raise NotImplementedError("This method should be overridden by subclasses")

    def create_target_monitor(self) -> Monitor:
        raise NotImplementedError("This method should be overridden by subclasses")


class MCSystemUnderObservation(SystemUnderObservation):
    def __init__(self, model_def: ModelDescription, name: str):
        self._model_def = model_def
        self._model, self.risk = build_model_and_risk(
            model_def,
            PremiseOptions(),
        )
        self.model_name = name
        self._ctr = ConditionalTraceGenerator(
            self._model, target_label=model_def.target_label
        )

    def get_states_and_transitions(self, add_label_to_state=False):
        return build_state_and_transition_list(
            self._model,
            self._model_def.target_label,
            all_transitions=True,
            add_label_to_state=add_label_to_state,
        )

    def generate_random_traces(self, observation_prefix: list, length: int, amount=1):
        return [
            self._ctr.generate_random_trace(observation_prefix, length)
            for _ in range(amount)
        ]

    def get_risk(self):
        return self.risk

    def create_target_monitor(self) -> Monitor:
        expr_manager = stormpy.ExpressionManager()  # type: ignore
        unfolder = stormpy.pomdp.create_observation_trace_unfolder(
            self._model, self.risk, expr_manager
        )
        ura = UnfoldingRiskAssessment(stormpy.Environment(), unfolder)  # type: ignore
        mon = Monitor(ura, 1000000)
        return mon
