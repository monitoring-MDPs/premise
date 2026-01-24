from dataclasses import dataclass

import tqdm
import time
import stormpy as sp
import stormpy.pomdp
from logging import getLogger

logger = getLogger(__name__)

@dataclass
class PremiseOptions:
    stormpy_environment : sp.Environment = sp.Environment()
    exact_arithmetic: bool = True
    promptness_deadline: int = 1000000000
    verbose: bool = False
    use_unfolding: bool = True
    restart_semantics: bool = False
    simulator_seed: int|None = None


class MonitorTimeOutException(Exception):
    """
    Exception raised when a monitor times out
    """
    pass


class Monitor:
    def __init__(self, riskassessor, deadline):
        self._risk_assessor = riskassessor
        self._deadline = deadline

    def initialize(self, observation : int):
        assert observation is not None
        self._risk_assessor.initialize(observation)

    def step(self, observation : int, compute_risk : bool = True):
        start_time = time.monotonic()
        self._risk_assessor.step(observation)
        if compute_risk:
            status, risk = self._risk_assessor.get_risk(deadline=self._deadline)
            if not status:
                raise MonitorTimeOutException
        else:
            risk = None
        end_time = time.monotonic()
        total_time = end_time - start_time
        if self._deadline and total_time * 1000 > self._deadline:
            raise MonitorTimeOutException
        return risk

    def dump_internal_data(self, path):
        """
        Dumps internal data from the risk assessor.
        """
        self._risk_assessor.dump_internal_data(path)


class UnfoldingRiskAssessment:
    """
    This class supports computing the risk based on unfolding the Markov model along the trace.
    """
    def __init__(self, stormpy_environment, unfolder):
        self._stormpy_env = stormpy_environment
        self._unfolder = unfolder
        self._mdp = None
        self._current_step = 0
        self._cache_until_compute = True # TODO set to true
        self._observation_cache = []
        self._prop = sp.parse_properties("Pmax=? [F \"_goal\"]")[0] if self._use_restart_semantics else sp.parse_properties("Pmax=? [F \"_goal\" || F \"_end\"]")[0]

    @property
    def _use_restart_semantics(self):
        ## TODO
        return self._unfolder.is_rejection_sampling_set()

    def initialize(self, observation):
        self._mdp = self._unfolder.reset(observation)
        self._current_step = 0

    def step(self, observation):
        """
        Makes a new step with the given observation.
        """
        if self._cache_until_compute:
            self._observation_cache.append(observation)
        else:
            self._mdp = self._unfolder.extend(observation)
        self._current_step += 1

    def dump_internal_data(self, path):
        """ Dumps the last created MDP model. """
        self.dump_model_to(path)

    def dump_model_to(self, base_path):
        """
        Dumps the last created MDP model to the given path.
        """
        path = base_path + f"-{self._current_step + 1}.drn"
        logger.info(f"Export MDP to {path}")
        sp.export_to_drn(self._mdp, path)

    def get_risk(self, deadline = None):
        """
        Computes the risk
        """
        if self._cache_until_compute:
            self._mdp = self._unfolder.extend(self._observation_cache)
            self._observation_cache = []
        sp.reset_timeout()
        if deadline:
            sp.set_timeout(int(deadline / 1000))
        try:
            result = sp.model_checking(self._mdp, self._prop, environment= self._stormpy_env, only_initial_states=True)
            sp.reset_timeout()
            risk = result.at(self._mdp.initial_states[0])
        except RuntimeError:
            print("TIMEOUT")
            logger.warning("Time out")
            sp.reset_timeout()
            return False, 0

        return True, risk


class FilterBasedRiskAssessment:
    """
    This class supports computing the risk based on a forward filtering.
    """
    def __init__(self, tracker):
        self._tracker = tracker

    def initialize(self, observation):
        result = self._tracker.reset(observation)

    def step(self, observation):
        self._tracker.reduce()
        passed = self._tracker.track(observation)
        if not passed:
            raise RuntimeError("Tracking failed")

    def get_risk(self):
        return True, self._tracker.obtain_current_risk()

    def dump_internal_data(self, path):
        """ Dumps internal data. Currently not implemented. """
        pass


def initialize_monitor(model, risk_structure, premise_options) -> Monitor:
    stormpy_environment = premise_options.stormpy_environment
    expr_manager = sp.ExpressionManager()
    if premise_options.use_unfolding:
        otu_options = sp.pomdp.ObservationTraceUnfolderOptions()
        otu_options.rejection_sampling = premise_options.restart_semantics
        unfolder = sp.pomdp.create_observation_trace_unfolder(model, risk_structure, expr_manager, options=otu_options) # restart_semantics= premise_options.restart_semantics
        riskassessor = UnfoldingRiskAssessment(stormpy_environment, unfolder)
    else:
        tracker = sp.pomdp.create_nondeterminstic_belief_tracker(model, premise_options.promptness_deadline, premise_options.promptness_deadline)
        riskassessor = FilterBasedRiskAssessment(tracker)
        tracker.set_risk(risk_structure)
    mon = Monitor(riskassessor, premise_options.promptness_deadline)
    return mon


def execute_monitor(trace_generator, monitor : Monitor, terminate_on_deadline = True, tqdm_bar = True):
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
                annotated_trace.append([(obs, None)])
                break
        annotated_trace.append((obs, risk))
        if tqdm_bar:
            pbar.update(1)
    if tqdm_bar:
        pbar.close()
    return annotated_trace
