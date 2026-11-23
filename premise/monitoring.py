from collections.abc import Iterable
import csv
from fractions import Fraction
import logging
import os
import os.path
from pathlib import Path
from statistics import mean, stdev
import time

import stormpy as sp
import stormpy.pomdp
import stormpy.simulator
from tqdm import tqdm

from premise import models, monitor, trace_generator, traces

logger = logging.getLogger(__name__)


def filtering(
    trace_generator,
    tracker,
    convex_reduction,
    stats_file,
    verbose,
    observation_valuations=None,
    terminate_on_deadline=True,
    deadline=None,
):
    """
    Runs forward filtering on a single trace.

    :param trace_generator: The (fixed-length) trace generator that spits out the observations
    :param tracker: The tracker that keeps track of the state estimation
    :param convex_reduction: If True, apply reduction after each step (recommended).
    :param stats_file: The file to output all the statistics to.
    :param verbose: If True, print some additional information
    :param observation_valuations: Used to print the high-level observations if verbose is set.
    :param deadline: How long can each step take (in ms).
    :param terminate_on_deadline: Can we abort if we took longer than deadline?
    :return: A tuple (completed, step_times) where completed indicates whether the whole trace was monitored
        and step_times contains the time (in seconds) taken for each step.
    """
    # TODO make stats files optional.
    observation = trace_generator.initialize()
    tracker.reset(observation)
    step_times = []

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
        i = 0
        while not trace_generator.finished():
            observation = trace_generator.step()
            if verbose:
                print(observation_valuations.to_string(observation, pretty=True))
                for belief in tracker.get_current_beliefs():
                    print(belief)
            start_time = time.monotonic()
            passed = tracker.track(observation)
            if not passed:
                writer.writerow([i, observation, 0, 0, 0, 0, 0, 0, 0, True])
                file.flush()
                return False, step_times
            risk = tracker.get_current_risk()
            end_time = time.monotonic()
            track_time = end_time - start_time
            start_time = time.monotonic()
            sizeBR = tracker.get_number_of_beliefs()
            if convex_reduction:
                tracker.reduce()
            end_time = time.monotonic()
            reduce_time = end_time - start_time
            timed_out = tracker.has_timed_out()
            total_time = track_time + reduce_time
            step_times.append(total_time)
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
                    tracker.get_number_of_beliefs(),
                    tracker.get_current_dimension(),
                    timed_out,
                ]
            )
            file.flush()
            i += 1
            if deadline and terminate_on_deadline and timed_out:
                return False, step_times
    return True, step_times


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
            mdp = unfolder.extend([observation])
            end_time = time.monotonic()
            unfold_time = end_time - start_time
            if dump_file_path is not None and (i + 1) % (int(trace_length / 10)) == 0:
                path = dump_file_path + f"-{i+1}.drn"
                logging.info(f"Export MDP to {path}")
                sp.io.export_to_drn(mdp, path)
            timeout = False
            start_time = time.monotonic()
            if do_model_checking:
                sp.developer.reset_timeout_alarm()
                sp.developer.set_timeout_alarm(int(deadline / 1000))

                try:
                    result = sp.model_checking(
                        mdp,
                        prop,
                        environment=stormpy_environment,
                        only_initial_states=True,
                    )
                    risk = result.get_value(mdp.initial_states[0])
                except RuntimeError:
                    timeout = True
                sp.developer.reset_timeout_alarm()
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
    def __init__(self, env, verbose=False, threshold=None):
        self.stormpy_environment = env
        self.verbose = verbose
        self.threshold = threshold


class ForwardFilteringOptions(StormConfigOptions):
    """
    Container to configure forward filtering.
    """

    def __init__(
        self,
        env=sp.Environment(),
        convex_hull_reduction=True,
        exact_arithmetic=True,
        threshold=None,
    ):
        super().__init__(env, threshold=threshold)
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
        env: sp.Environment | None = sp.Environment(),
        exact_arithmetic=True,
        use_rejection_sampling=True,
        conditional_method=sp.environments.ConditionalAlgorithmSetting.DEFAULT,
        model_checking_method=None,
        custom_str=None,
        export_models_path=None,
        threshold=None,
        force_exact=False,
    ):
        super().__init__(env, threshold=threshold)
        self.exact_arithmetic = exact_arithmetic
        self._numstr = custom_str
        self.export_models_path = export_models_path
        self.use_rejection_sampling = use_rejection_sampling
        self.conditional_method = (
            self._map_conditional_method(conditional_method)
            if type(conditional_method) == str
            else conditional_method
        )
        self.force_exact = force_exact
        if model_checking_method is None:
            self.model_checking_method = None
        else:
            self.model_checking_method = (
                self._map_mc_method(model_checking_method)
                if type(model_checking_method) == str
                else model_checking_method
            )

    @staticmethod
    def _map_conditional_method(method: str) -> sp.environments.ConditionalAlgorithmSetting:
        if method in ["default", "rejection"]:
            return sp.environments.ConditionalAlgorithmSetting.DEFAULT
        elif method == "restart":
            return sp.environments.ConditionalAlgorithmSetting.RESTART
        elif method == "bisection":
            return sp.environments.ConditionalAlgorithmSetting.BISECTION
        elif method == "bisection-advanced":
            return sp.environments.ConditionalAlgorithmSetting.BISECTION_ADVANCED
        elif method == "pi":
            return sp.environments.ConditionalAlgorithmSetting.POLICY_ITERATION
        else:
            raise ValueError(f"Unknown conditional method: {method}")

    @staticmethod
    def _map_mc_method(method: str) -> sp.environments.MinMaxMethod:
        if method == "value_iteration":
            return sp.environments.MinMaxMethod.VALUE_ITERATION
        elif method == "policy_iteration":
            return sp.environments.MinMaxMethod.POLICY_ITERATION
        elif method == "sound_value_iteration":
            return sp.environments.MinMaxMethod.SOUND_VALUE_ITERATION
        elif method == "optimistic_value_iteration":
            return sp.environments.MinMaxMethod.OPTIMISTIC_VALUE_ITERATION
        elif method == "":
            return None
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

            rej = (
                "rej"
                if self.use_rejection_sampling
                else f"vio-{self.conditional_method}"
            )
            if self.conditional_method in [
                sp.environments.ConditionalAlgorithmSetting.BISECTION_ADVANCED,
                sp.environments.ConditionalAlgorithmSetting.BISECTION,
            ]:
                rej += f"-{self.model_checking_method}"

            numstr = f"{numstr}-{rej}"

        return f"unf-{numstr}"


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
    stats_path=Path("./stats"),
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
    logger.info(f"Starting monitor with arguments: {locals()}")
    start_time = time.monotonic()

    stats_folder = stats_path / f"{model_id}-{options.method_id}/"
    if not os.path.isdir(stats_folder):
        os.makedirs(stats_folder)
    else:
        logger.info(f"We are writing to an existing folder '{stats_folder}'.")

    use_forward_filtering = isinstance(options, ForwardFilteringOptions)
    use_unfolding = isinstance(options, UnfoldingOptions)
    if not use_forward_filtering and not use_unfolding:
        raise RuntimeError("Unknown type of options, method cannot be deduced")
    assert not (use_forward_filtering and use_unfolding)

    logger.info("Parse MDP representation...")
    model, risk_assessment = models.build_model_and_risk(
        models.ModelDescription(path, constants, risk_property), options
    )

    stormpy_environment = options.stormpy_environment
    if use_forward_filtering:
        logger.info("Initialize tracker...")
        tracker = sp.pomdp.create_nondeterministic_belief_tracker(
            model,
            reduction_timeout=promptness_deadline,
            track_timeout=promptness_deadline,
        )
        tracker.set_risk(risk_assessment)
        mon = None
    else:
        assert use_unfolding
        assert isinstance(options, UnfoldingOptions)
        logger.info("Initialize unfolder...")
        expr_manager = sp.expressions.ExpressionManager()
        unfolder = sp.pomdp.create_observation_trace_unfolder(
            model,
            risk_assessment,
            expr_manager,
            use_restart_semantics=options.use_rejection_sampling,
        )

        ura = monitor.UnfoldingRiskAssessment(
            stormpy_environment, unfolder, options.threshold
        )

        mon = monitor.Monitor(ura, promptness_deadline)

    initialize_time = time.monotonic() - start_time

    if isinstance(simulator_seed, Iterable):
        simulator_seed_range = simulator_seed
    else:
        simulator_seed_range = range(simulator_seed, simulator_seed + 1)

    times_taken = {}
    time_per_step = {}

    for seed in tqdm(simulator_seed_range):
        logger.info(
            f"Running monitor for seed ({len(times_taken)}/{len(simulator_seed_range)}) {seed}..."
        )

        stg = trace_generator.make_simulation_wrapper(model, trace_length, seed)
        cache_path = Path(f"{stats_folder}").parent / f"simulator-caches-{model_id}"
        cache_path.mkdir(parents=True, exist_ok=True)
        stg = trace_generator.FileCachedSimulationTraceGenerator(
            stg,  # type: ignore
            f"{cache_path}/simulator-cache-{seed}",
        )
        logger.info("Restart simulator...")

        stats_file = f"{stats_folder}/stats-{model_id}-{options.method_id}-{seed}.csv"

        if use_forward_filtering:
            _, step_times = filtering(
                stg,
                tracker,
                options.convex_hull_reduction,
                stats_file,
                deadline=promptness_deadline,
                verbose=verbose,
                observation_valuations=model.observation_valuations,
            )
            time_per_step[seed] = step_times
            times_taken[seed] = sum(step_times)
        else:
            assert use_unfolding and mon is not None
            annotated_trace = monitor.execute_monitor(stg, mon, tqdm_bar=False)
            time_per_step[seed] = mon.risk_times.copy()
            times_taken[seed] = sum(mon.risk_times)
            mon.risk_times.clear()
            trace_mapper = traces.TraceMapper(model)
            trace_file = (
                f"{stats_folder}/trace-{model_id}-{options.method_id}-{seed}.csv"
            )
            traces.export_annotated_high_level(
                trace_mapper.annotated_trace_to_highlevel(annotated_trace),
                model,
                trace_file,
            )

    with open(os.path.join(stats_folder, "stats.out"), "w") as file:
        file.write(f"states={model.nr_states}\n")
        file.write(f"transitions={model.nr_transitions}\n")
        file.write(f"init_time={initialize_time}\n")
        file.write(f"promptness_deadline={promptness_deadline}\n")
        file.write(f"options={options}\n")
        file.write(f"max_time={max(times_taken.values())}\n")
        file.write(f"min_time={min(times_taken.values())}\n")
        file.write(f"avg_time={sum(times_taken.values())/len(times_taken)}\n")
        file.write(
            f"worst_5_seeds={sorted(times_taken.items(), key=lambda item: item[1], reverse=True)[:5]}\n"
        )
        file.write(
            f"best_5_seeds={sorted(times_taken.items(), key=lambda item: item[1])[:5]}\n"
        )
        file.write(f"all_times={times_taken}\n")
        file.write(f"time_per_step={time_per_step}\n")


def create_benchmark_models(
    path, constants, risk_property, options, trace_length, seeds, model_path
):
    logger.info("Parse MDP representation")
    model, risk_assessment = models.build_model_and_risk(
        models.ModelDescription(path, constants, risk_property), options
    )
    sp.io.export_to_drn(
        model,
        model_path + f"full-{Path(path).stem}" + ".drn",
    )
    fraction_risks = [Fraction(str(risk)) for risk in risk_assessment]
    avg_risk = mean(fraction_risks)
    std_risk = stdev(fraction_risks)
    logger.info(f"Model has {model.nr_states} states")

    logger.info("Initialize unfolder")
    stormpy_environment = options.stormpy_environment
    expr_manager = sp.expressions.ExpressionManager()
    unfolder = sp.pomdp.create_observation_trace_unfolder(
        model, risk_assessment, expr_manager, use_restart_semantics=False
    )

    logger.info("Looping over seeds to create benchmark models")

    marginals = {}
    for seed in tqdm(seeds):
        logger.info(f"Creating benchmark model for seed {seed}")
        stg = trace_generator.make_simulation_wrapper(model, trace_length, seed)

        observations = [stg.initialize()]

        for _ in range(trace_length):
            observations.append(stg.step())

        logger.info(f"Creating unrolled model for seed {seed}")
        mdp = unfolder.transform(observations)
        prop = sp.parse_properties(f'Pmax=? [F "_end"]')[0]
        result = sp.model_checking(
            mdp,
            prop,
            environment=stormpy_environment,
            only_initial_states=True,
        )
        marginal = result.get_value(mdp.initial_states[0])
        marginals[seed] = marginal
        print(f"Seed {seed}, marginal: {marginal}")
        if marginal == 0:
            raise RuntimeError(
                "The marginal risk of reaching the condition is 0, incorrect benchmark model."
            )

        sp.io.export_to_drn(
            mdp,
            model_path
            + f"{Path(path).stem}"
            + (f"-{seed}" if len(seeds) > 1 else "")
            + ".drn",
        )

    with open(model_path + Path(path).stem + "-stats.out", "w") as file:
        file.write(f"states={model.nr_states}\n")
        file.write(f"transitions={model.nr_transitions}\n")
        file.write(f"avg_risk= ({float(avg_risk)}) {avg_risk}\n")
        file.write(f"std_risk= ({float(std_risk)}) {std_risk}\n")
        file.write("marginals:\n")
        for seed, marginal in marginals.items():
            file.write(f"  seed {seed}: ({float(marginal)}) {marginal}\n")
