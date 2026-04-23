from dataclasses import dataclass

import tqdm
import time
import stormpy as sp
import stormpy.pomdp
from logging import getLogger

logger = getLogger(__name__)


@dataclass
class PremiseOptions:
    stormpy_environment: sp.Environment = sp.Environment()
    exact_arithmetic: bool = True
    promptness_deadline: int = 1000000000
    verbose: bool = False
    use_unfolding: bool = True
    use_rejection: bool = True


class MonitorTimeOutException(Exception):
    pass


class UnfoldingRiskAssessment:
    def __init__(
        self,
        stormpy_environment,
        unfolder,
        threshold=None,
    ):
        self._stormpy_env = stormpy_environment
        self._unfolder = unfolder
        self._mdp = None
        self._current_step = 0
        use_conditional_method = not unfolder.is_restart_semantics_set()

        self._threshold = threshold
        if threshold is not None:
            threshold_prop = f"max<={threshold}"
        else:
            threshold_prop = "max=?"

        if use_conditional_method:
            self._prop = sp.parse_properties(
                f'P{threshold_prop} [F "_goal" || F "_end"]'
            )[0]
        else:
            self._prop = sp.parse_properties(f'P{threshold_prop} [F "_goal"]')[0]
        print(self._prop)

    def initialize(self, observation):
        self._mdp = self._unfolder.reset(observation)

    def step(self, observation, dump_model_to=None):
        """
        Makes a new step with the given observation.
        """
        self._mdp = self._unfolder.extend([observation])
        if dump_model_to is not None:
            path = dump_model_to + f"-{self._current_step + 1}.drn"
            logger.info(f"Export MDP to {path}")
            sp.export_to_drn(self._mdp, path)

    def get_risk(self, deadline=None):
        """
        Computes the risk
        """
        sp.reset_timeout()
        if deadline:
            pass
            # sp.set_timeout(int(deadline / 1000))
        try:
            result = sp.model_checking(
                self._mdp,
                self._prop,
                environment=self._stormpy_env,
                only_initial_states=True,
            )
            sp.reset_timeout()
            risk = result.at(self._mdp.initial_states[0])
        except RuntimeError:
            print("What")
            sp.reset_timeout()
            logger.warning("Time out")
            return False, 0

        return True, risk


class FilterBasedRiskAssessment:
    def __init__(self, tracker):
        self._tracker = tracker

    def initialize(self, observation):
        self._tracker.reset(observation)

    def step(self, observation):
        self._tracker.reduce()
        passed = self._tracker.track(observation)
        if not passed:
            raise RuntimeError("Tracking failed")

    def get_risk(self, deadline=None):
        return self._tracker.obtain_current_risk()


class Monitor:
    def __init__(
        self,
        riskassessor: UnfoldingRiskAssessment | FilterBasedRiskAssessment,
        deadline,
    ):
        self._risk_assessor = riskassessor
        self._deadline = deadline
        self.risk_times = []

    def initialize(self, observation, compute_risk=True):
        self._risk_assessor.initialize(observation)

    def step(self, observation, compute_risk=True):
        start_time = time.monotonic()
        self._risk_assessor.step(observation)
        if compute_risk:
            start_risk_time = time.monotonic()
            status, risk = self._risk_assessor.get_risk(self._deadline)
            end_risk_time = time.monotonic()
            self.risk_times.append(end_risk_time - start_risk_time)
            if not status:
                raise MonitorTimeOutException
        else:
            risk = None
        end_time = time.monotonic()
        total_time = end_time - start_time
        if self._deadline and total_time * 1000 > self._deadline:
            raise MonitorTimeOutException
        return risk


def initialize_monitor(model, risk_structure, premise_options) -> Monitor:
    stormpy_environment = premise_options.stormpy_environment
    expr_manager = sp.ExpressionManager()
    if premise_options.use_unfolding:
        otu_options = sp.pomdp.ObservationTraceUnfolderOptions()
        otu_options.rejection_sampling = premise_options.rejection_sampling
        unfolder = sp.pomdp.create_observation_trace_unfolder(
            model, risk_structure, expr_manager, options=otu_options
        )
        ura = UnfoldingRiskAssessment(stormpy_environment, unfolder)
    else:
        raise NotImplementedError("Something is missing here.")
    mon = Monitor(ura, premise_options.promptness_deadline)
    return mon


def execute_monitor(
    trace_generator, monitor: Monitor, terminate_on_deadline=True, tqdm_bar=True
):
    annotated_trace = []
    obs = trace_generator.initialize()
    monitor.initialize(obs)
    if tqdm_bar:
        pbar = tqdm.tqdm(total=trace_generator.max_length)
    while not trace_generator.finished():
        obs = trace_generator.step()
        try:
            risk = monitor.step(obs)
        except MonitorTimeOutException:
            if terminate_on_deadline:
                annotated_trace.append((obs, None))
                break
        annotated_trace.append((obs, risk))
        if tqdm_bar:
            pbar.update(1)
    if tqdm_bar:
        pbar.close()
    return annotated_trace
