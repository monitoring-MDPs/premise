from collections.abc import Iterable
import csv
import logging
import os
import os.path
import re
import stat
import time

import stormpy as sp
import stormpy.pomdp
import stormpy.simulator
from tqdm import tqdm

import models
import monitor
import trace_generator
import traces

logger = logging.getLogger(__name__)


def filtering(
    stormpy_environment,
    simulator,
    tracker,
    trace_length,
    convex_reduction,
    stats_file,
    verbose,
    observation_valuations=None,
    terminate_on_deadline=True,
    deadline=None,
):
    """

    :param simulator: The simulator that spits out the observations
    :param tracker: The tracker that keeps track of the state estimation
    :param trace_length: How many steps to take.
    :param convex_reduction: If True, apply reduction after each step (recommended).
    :param stats_file: The file to output all the statistics to.
    :param verbose: If True, print some additional information
    :param observation_valuations:
    :param deadline: How long can each step take.
    :param terminate_on_deadline: Can we abort if we took longer than deadline?
    :return:
    """
    # TODO make stats files optional.
    observation, _ = simulator.restart()
    tracker.reset(observation)

    with open(stats_file, "w") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "Index",
                "Observation",
                "Risk",
                "TrackTime",
                "ReduceTime",
                "TotalTime",
                "NrBeliefsBR",
                "NrBeliefsAR",
                "Dimension",
                "TimedOut",
            ]
        )
        iterator = tqdm(range(trace_length))
        for i in iterator:
            observation, _, _ = simulator.random_step()
            if verbose:
                hl_obs = observation_valuations.get_string(observation, pretty=True)[
                    1:-1
                ].replace("\t", " ")
                for belief in tracker.obtain_beliefs():
                    print(belief)
            start_time = time.monotonic()
            passed = tracker.track(observation)
            if not passed:
                writer.writerow([i, observation, 0, 0, 0, 0, 0, 0, 0, True])
                file.flush()

                iterator.close()
                return False
            risk = tracker.obtain_current_risk()
            end_time = time.monotonic()
            track_time = end_time - start_time
            start_time = time.monotonic()
            sizeBR = tracker.size()
            if convex_reduction:
                tracker.reduce()
            end_time = time.monotonic()
            reduce_time = end_time - start_time
            timed_out = tracker.reduction_timed_out()
            total_time = track_time + reduce_time
            if deadline is not None and total_time * 1000 > deadline:
                timed_out = True
            writer.writerow(
                [
                    i,
                    observation,
                    risk,
                    track_time,
                    reduce_time,
                    total_time,
                    sizeBR,
                    tracker.size(),
                    tracker.dimension(),
                    timed_out,
                ]
            )
            file.flush()
            if deadline and terminate_on_deadline and timed_out:
                iterator.close()
                return False
        iterator.close()
    return True


def unfolding(
    stormpy_environment,
    simulator,
    unfolder,
    trace_length,
    stats_file,
    deadline=None,
    terminate_on_deadline=True,
    dump_file_path=None,
    do_model_checking=True,
):
    """

    :param stormpy_environment: The stormpy environment to use
    :param simulator: The simulator that spits out the observations
    :param unfolder: The tracker that keeps track of the state estimation
    :param trace_length: How many steps to take.
    :param stats_file: The file to output all the statistics to.
    :param deadline: The deadline for every computation step
    :param terminate_on_deadline: Should we abort the trace if we exceeded computation time
    :param dump_file_path: Path to dump model files. None to disable dumping them.
    :param do_model_checking: Whether to actually verify the unfolding (Set to false if one is only interested in generating models)
    :return:
    """
    # TODO make the stats_file optional
    observation, _, _ = simulator.restart()
    unfolder.reset(observation)
    prop = sp.parse_properties('Pmax=? [F "_goal"]')[0]

    with open(stats_file, "w") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "Index",
                "Observation",
                "Risk",
                "UnfTime",
                "McTime",
                "TotalTime",
                "MdpStates",
                "MdpTransitions",
                "TimedOut",
            ]
        )
        for i in tqdm(range(trace_length)):
            observation, _, _ = simulator.random_step()
            start_time = time.monotonic()
            mdp = unfolder.extend(observation)
            end_time = time.monotonic()
            unfold_time = end_time - start_time
            if dump_file_path is not None and (i + 1) % (int(trace_length / 10)) == 0:
                path = dump_file_path + f"-{i+1}.drn"
                logging.info(f"Export MDP to {path}")
                stormpy.export_to_drn(mdp, path)
            timeout = False
            start_time = time.monotonic()
            if do_model_checking:
                stormpy.reset_timeout()
                stormpy.set_timeout(int(deadline / 1000))

                try:
                    result = stormpy.model_checking(
                        mdp,
                        prop,
                        environment=stormpy_environment,
                        only_initial_states=True,
                    )
                    risk = result.at(mdp.initial_states[0])
                except RuntimeError:
                    timeout = True
                stormpy.reset_timeout()
                end_time = time.monotonic()
                mc_time = end_time - start_time
                total_time = unfold_time + mc_time
                if deadline and total_time * 1000 > deadline:
                    timeout = True
                writer.writerow(
                    [
                        i,
                        observation,
                        risk,
                        unfold_time,
                        mc_time,
                        total_time,
                        mdp.nr_states,
                        mdp.nr_transitions,
                        timeout,
                    ]
                )
            else:
                writer.writerow(
                    [
                        i,
                        observation,
                        "NA",
                        unfold_time,
                        "NA",
                        "NA",
                        mdp.nr_states,
                        mdp.nr_transitions,
                        timeout,
                    ]
                )
            file.flush()
            if terminate_on_deadline and timeout:
                return False
    return True


class StormConfigOptions:
    def __init__(self, env, verbose=False):
        self.stormpy_environment = env
        self.verbose = verbose


class ForwardFilteringOptions(StormConfigOptions):
    """
    Container to configure forward filtering.
    """

    def __init__(
        self, env=sp.Environment(), convex_hull_reduction=True, exact_arithmetic=True
    ):
        super().__init__(env)
        self.convex_hull_reduction = convex_hull_reduction
        self.exact_arithmetic = exact_arithmetic

    def __str__(self):
        return f"ForwardFiltering(exact_numbers={self.exact_arithmetic},chreduction={self.convex_hull_reduction})"

    @property
    def method_id(self):
        redstr = "ch" if self.convex_hull_reduction else "nr"
        numstr = "ea" if self.exact_arithmetic else "fl"
        return "ff-{}-{}".format(redstr, numstr)


class UnfoldingOptions(StormConfigOptions):
    """
    Container to configure unfolding.
    """

    def __init__(
        self,
        env=sp.Environment(),
        exact_arithmetic=True,
        use_rejection_sampling=True,
        conditional_method=sp.ConditionalAlgorithmSetting.default,
        model_checking_method=sp.MinMaxMethod.value_iteration,
        custom_str=None,
        export_models_path=None,
    ):
        super().__init__(env)
        self.exact_arithmetic = exact_arithmetic
        self._numstr = custom_str
        self.export_models_path = export_models_path
        self.use_rejection_sampling = use_rejection_sampling
        self.conditional_method = (
            self._map_conditional_method(conditional_method)
            if type(conditional_method) == str
            else conditional_method
        )
        self.model_checking_method = (
            self._map_mc_method(model_checking_method)
            if type(model_checking_method) == str
            else model_checking_method
        )

    @staticmethod
    def _map_conditional_method(method: str) -> sp.ConditionalAlgorithmSetting:
        if method == "default":
            return sp.ConditionalAlgorithmSetting.default
        elif method == "restart":
            return sp.ConditionalAlgorithmSetting.restart
        elif method == "bisection":
            return sp.ConditionalAlgorithmSetting.bisection
        elif method == "bisection-advanced":
            return sp.ConditionalAlgorithmSetting.bisection_advanced
        elif method == "pi":
            return sp.ConditionalAlgorithmSetting.policy_iteration
        else:
            raise ValueError(f"Unknown conditional method: {method}")

    @staticmethod
    def _map_mc_method(method: str) -> sp.MinMaxMethod:
        if method == "value_iteration":
            return sp.MinMaxMethod.value_iteration
        elif method == "policy_iteration":
            return sp.MinMaxMethod.policy_iteration
        elif method == "sound_value_iteration":
            return sp.MinMaxMethod.sound_value_iteration
        elif method == "optimistic_value_iteration":
            return sp.MinMaxMethod.optimistic_value_iteration
        else:
            raise ValueError(f"Unknown MC method: {method}")

    def __str__(self):
        return f"Unfolding(exact_numbers={self.exact_arithmetic},use_rejection_sampling={self.use_rejection_sampling},conditional_method={self.conditional_method},custom_str={self._numstr})"

    @property
    def method_id(self):
        if self._numstr is not None:
            numstr = self._numstr
        else:
            numstr = "ea" if self.exact_arithmetic else "fl"

        rej = "rej" if self.use_rejection_sampling else f"vio-{self.conditional_method}"
        if self.conditional_method in [
            sp.ConditionalAlgorithmSetting.bisection_advanced,
            sp.ConditionalAlgorithmSetting.bisection,
        ]:
            rej += f"-{self.model_checking_method}"

        return f"unf-{numstr}-{rej}"


def unrolled_model_path(options, model_id, seed):
    if options.export_models_path is not None:
        models_folder = f"{options.export_models_path}/{model_id}"
        if not os.path.isdir(models_folder):
            os.makedirs(models_folder)
        unrolled_drn_file_prefix = f"{models_folder}-{seed}"
        if not os.path.isdir(unrolled_drn_file_prefix):
            os.makedirs(unrolled_drn_file_prefix)
        return unrolled_drn_file_prefix
    else:
        return None


def run_monitor(
    path,
    risk_property,
    constants,
    trace_length,
    options: UnfoldingOptions | ForwardFilteringOptions,
    verbose=False,
    simulator_seed: list[int] | int = 0,
    promptness_deadline=10000,
    model_id="no_id_given",
):
    """

    :param path: The path the the model file
    :param risk_property: The property that describes the risk
    :param constants: Values for constants that appear in the model.
    :param trace_length: How long should the traces be
    :param options: Options to configure the monitoring.
    :param verbose: Should we print
    :param simulator_seed: A range of seeds we use for the simulator.
    :param promptness_deadline:
    :param model_id: A name for creating good stats files.
    :return:
    """
    start_time = time.monotonic()
    use_forward_filtering = isinstance(options, ForwardFilteringOptions)
    use_unfolding = isinstance(options, UnfoldingOptions)
    if not use_forward_filtering and not use_unfolding:
        raise RuntimeError("Unknown type of options, method cannot be deduced")
    assert not (use_forward_filtering and use_unfolding)

    logger.info("Parse MDP representation...")
    model, risk_assessment = models.build_model_and_risk(
        models.ModelDescription(path, constants, risk_property), options
    )

    if use_forward_filtering:
        logger.info("Initialize tracker...")
        tracker = stormpy.pomdp.create_nondeterminstic_belief_tracker(
            model, promptness_deadline, promptness_deadline
        )
        tracker.set_risk(risk_assessment)
    else:
        assert use_unfolding
        assert isinstance(options, UnfoldingOptions)
        logger.info("Initialize unfolder...")
        stormpy_environment = options.stormpy_environment
        expr_manager = stormpy.ExpressionManager()
        unfolding_options = stormpy.pomdp.ObservationTraceUnfolderOptions()
        unfolding_options.rejection_sampling = options.use_rejection_sampling
        stormpy_environment.model_checker_environment.conditional_algorithm = (
            options.conditional_method
        )
        stormpy_environment.solver_environment.minmax_solver_environment.method = (
            options.model_checking_method
        )
        unfolder = stormpy.pomdp.create_observation_trace_unfolder(
            model, risk_assessment, expr_manager, unfolding_options
        )

        ura = monitor.UnfoldingRiskAssessment(
            stormpy_environment,
            unfolder,
            None if options.use_rejection_sampling else options.conditional_method,
        )

        mon = monitor.Monitor(ura, promptness_deadline)

    initialize_time = time.monotonic() - start_time
    stats_folder = f"stats/{model_id}-{options.method_id}/"
    if not os.path.isdir(stats_folder):
        os.makedirs(stats_folder)
    else:
        raise RuntimeWarning(f"We are writing to an existing folder '{stats_folder}'.")

    logger.info("Initialize simulator...")
    simulator = sp.simulator.create_simulator(model)

    if isinstance(simulator_seed, Iterable):
        simulator_seed_range = simulator_seed
    else:
        simulator_seed_range = range(simulator_seed, simulator_seed + 1)

    times_taken = {}
    for seed in tqdm(simulator_seed_range):
        if seed in [
            4458661313036743149,
            17148839251503414911,
            3471586787901513327,
            2396310440082695277,
            17752478338837792148,
            8786698114815218908,
            15799920621583114172,
            8327441564423680047,
            1699671342762650436,
        ]:
            logger.info(f"Skipping seed {seed} due to known issues.")
            continue
        stg = trace_generator.make_simulation_wrapper(model, trace_length, seed)
        logger.info("Restart simulator...")

        stats_file = f"{stats_folder}/stats-{model_id}-{options.method_id}-{seed}.csv"

        if use_forward_filtering:
            filtering(
                stormpy_environment,
                simulator,
                tracker,
                trace_length,
                options.convex_hull_reduction,
                stats_file,
                deadline=promptness_deadline,
                verbose=verbose,
                observation_valuations=model.observation_valuations,
            )
        else:
            assert use_unfolding
            # unrolled_model_path(options, model_id, seed)
            t = time.monotonic()
            annotated_trace = monitor.execute_monitor(stg, mon)
            times_taken[seed] = time.monotonic() - t
            trace_mapper = traces.TraceMapper(model)
            trace_file = (
                f"{stats_folder}/trace-{model_id}-{options.method_id}-{seed}.csv"
            )
            traces.export_annotated_high_level(
                trace_mapper.annotated_trace_to_highlevel(annotated_trace),
                model,
                trace_file,
            )

            # unfolding(stormpy_environment, simulator, unfolder, trace_length, stats_file, deadline=promptness_deadline, dump_file_path=unrolled_drn_file_prefix)

    with open(os.path.join(stats_folder, "stats.out"), "w") as file:
        file.write(f"states={model.nr_states}\n")
        file.write(f"transitions={model.nr_transitions}\n")
        file.write(f"init_time={initialize_time}\n")
        file.write(f"promptness_deadline={promptness_deadline}\n")
        file.write(f"max_time={max(times_taken.values())}\n")
        file.write(f"min_time={min(times_taken.values())}\n")
        file.write(f"avg_time={sum(times_taken.values())/len(times_taken)}\n")
