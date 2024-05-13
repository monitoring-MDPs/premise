from dataclasses import dataclass

import tqdm
import time
import stormpy as sp
import stormpy.pomdp
import stormpy.simulator
from logging import getLogger

logger = getLogger(__name__)

@dataclass
class PremiseOptions:
    stormpy_environment : sp.Environment = sp.Environment()
    exact_arithmetic: bool = True
    promptness_deadline: int = 1000000000
    verbose: bool = False

def execute_monitor(trace_generator, monitor, terminate_on_deadline = True, tqdm_bar = True):
    annotated_trace = []
    obs = trace_generator.initialize()
    monitor.initialize(obs)
    if tqdm_bar:
        pbar = tqdm(total = trace_generator.max_length)
    while not trace_generator.finished():
        obs = trace_generator.step()
        try:
            time_out, risk = monitor.step(obs)
        except MonitorTimeOutException:
            if terminate_on_deadline:
                annotated_trace.append([(obs, None)])
                break
        annotated_trace.append((obs, risk))
        if tqdm_bar:
            pbar.update(1)
    if tqdm_bar:
        pbar.close()
    return annotated_trace


class MonitorTimeOutException(Exception):
    """"""
    pass

class Monitor:
    def __init__(self, riskassessor, deadline):
        self._risk_assessor = riskassessor
        self._deadline = deadline

    def initialize(self, observation, compute_risk = True):
        self._risk_assessor.initialize(observation)


    def step(self, observation, compute_risk = True):
        start_time = time.monotonic()
        self._risk_assessor.step(observation)
        if compute_risk:
            status, risk = self._risk_assessor.get_risk()
            if not status:
                raise MonitorTimeOutException
        else:
            risk = None
        end_time = time.monotonic()
        total_time = end_time - start_time
        if self._deadline and total_time * 1000 > self._deadline:
            raise MonitorTimeOutException
        return risk

class FilterBasedRiskAssessment:
    def __init__(self, tracker):
        self._tracker = tracker

    def initialize(self, observation):
        self._tracker

    def step(self, observation):
        passed = self._tracker.track(observation)
        if not passed:
            return False

    def get_risk(self):
        pass


class UnfoldingRiskAssessment:
    def __init__(self, stormpy_environment, unfolder):
        self._stormpy_env = stormpy_environment
        self._unfolder = unfolder
        self._mdp = None
        self._current_step = 0
        self._prop = sp.parse_properties("Pmax=? [F \"_goal\"]")[0]

    def initialize(self, observation):
        self._mdp = self._unfolder.reset(observation)

    """
    Makes a new step with the given observation.
    """
    def step(self, observation, dump_model_to = None):
        self._mdp = self._unfolder.extend(observation)
        if dump_model_to is not None:
            path = dump_model_to + f"-{self._current_step + 1}.drn"
            logger.info(f"Export MDP to {path}")
            sp.export_to_drn(self._mdp, path)

    """
    Computes the risk
    """
    def get_risk(self, deadline = None):
        sp.reset_timeout()
        if deadline:
            sp.set_timeout(int(deadline / 1000))
        try:
            result = sp.model_checking(self._mdp, self._prop, environment= self._stormpy_env, only_initial_states=True)
            risk = result.at(self._mdp.initial_states[0])
        except RuntimeError:
            print("What")
            logger.warning("Time out")
            return False, 0
        sp.reset_timeout()
        return True, risk


def initialize_monitor(model, risk_structure, premise_options) -> Monitor:
    stormpy_environment = premise_options.stormpy_environment
    expr_manager = sp.ExpressionManager()
    unfolder = sp.pomdp.create_observation_trace_unfolder(model, risk_structure, expr_manager)
    ura = UnfoldingRiskAssessment(stormpy_environment, unfolder)
    mon = Monitor(ura, premise_options.promptness_deadline)
    return mon


