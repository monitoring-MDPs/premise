from collections.abc import Iterable
import csv
import logging
import os
import os.path
import time

import stormpy as sp
import stormpy.pomdp
import stormpy.simulator
from tqdm import tqdm

logger = logging.getLogger(__name__)

def build_model(program, formula, exact_arithmetic):
    """
    Takes a model and a formula that describes the bad event.

    :param program: PRISM program
    :param formula: a PCTL specification for a bad event
    :param exact_arithmetic: Flag for using exact arithmetic.
    :return: A sparse MDP
    """
    options = sp.BuilderOptions([formula])
    options.set_build_state_valuations()
    options.set_build_observation_valuations()
    options.set_build_choice_labels()
    options.set_build_all_labels()
    logger.debug("Start building the MDP")
    if exact_arithmetic:
        return sp.build_sparse_exact_model_with_options(program, options)
    else:
        return sp.build_sparse_model_with_options(program, options)


def analyse_model(model, prop):
    """
    Analyse the MDP with respect to the given property. Assume full observability.
    :param model:
    :param prop:
    :return:
    """
    return stormpy.model_checking(model, prop.raw_formula, force_fully_observable=True)


def filtering(simulator, tracker, trace_length, convex_reduction, stats_file, verbose, observation_valuations = None, terminate_on_deadline = True, deadline=None):
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
    #TODO make stats files optional.
    observation, _ = simulator.restart()
    tracker.reset(observation)

    with open(stats_file, 'w') as file:
        writer = csv.writer(file)
        writer.writerow(["Index", "Observation", "Risk", "TrackTime", "ReduceTime","TotalTime", "NrBeliefsBR", "NrBeliefsAR", "Dimension", "TimedOut"])
        iterator = tqdm(range(trace_length))
        for i in iterator:
            observation, _ = simulator.random_step()
            if verbose:
                hl_obs = observation_valuations.get_string(observation, pretty=True)[1:-1].replace("\t", " ")
                print(f"Observe {hl_obs}")
                for belief in tracker.obtain_beliefs():
                    print(belief)
            start_time = time.monotonic()
            passed = tracker.track(observation)
            if not passed:
                writer.writerow(
                    [i, observation, 0, 0, 0, 0, 0, 0,
                     0,True])
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
            total_time = track_time+reduce_time
            if deadline is not None and total_time * 1000 > deadline:
                timed_out = True
            writer.writerow([i, observation, risk, track_time, reduce_time, total_time, sizeBR, tracker.size(), tracker.dimension(), timed_out])
            file.flush()
            if deadline and terminate_on_deadline and timed_out:
                iterator.close()
                return False
        iterator.close()
    return True

def unfolding(simulator, unfolder, trace_length, stats_file, deadline=None, terminate_on_deadline = True, use_ovi=False):
    """

    :param simulator: The simulator that spits out the observations
    :param tracker: The tracker that keeps track of the state estimation
    :param trace_length: How many steps to take.
    :param stats_file: The file to output all the statistics to.
    :param deadline: The deadline for every computation step
    :param terminate_on_deadline: Should we abort the trace if we exceeded computation time
    :param use_ovi: Whether to use OVI or not.
    :return:
    """
    #TODO make the stats_file optional
    observation, _ = simulator.restart()
    unfolder.reset(observation)
    env = stormpy.Environment()
    if use_ovi:
        env.solver_environment.minmax_solver_environment.method = stormpy.MinMaxMethod.optimistic_value_iteration
        env.solver_environment.minmax_solver_environment.precision = stormpy.Rational("0.01")

    with open(stats_file, 'w') as file:
        writer = csv.writer(file)
        writer.writerow(["Index", "Observation", "Risk", "UnfTime", "McTime", "TotalTime", "MdpStates", "MdpTransitions", "TimedOut"])
        for i in tqdm(range(trace_length)):
            observation, _ = simulator.random_step()
            start_time = time.monotonic()
            mdp = unfolder.extend(observation)
            end_time = time.monotonic()
            unfold_time = end_time - start_time
            #stormpy.export_to_drn(mdp,"unrolling.out")
            prop = sp.parse_properties("Pmax=? [F \"_goal\"]")[0]
            #
            #mdpl = stormpy.build_model_from_drn("unrolling.out")
            timeout = False
            start_time = time.monotonic()
            stormpy.reset_timeout()
            stormpy.set_timeout(int(deadline/1000))
            stormpy.install_signal_handlers()
            try:
                result = stormpy.model_checking(mdp, prop, environment=env, only_initial_states=True)
                risk = result.at(0)
            except RuntimeError:
                timeout = True
            stormpy.reset_timeout()
            end_time = time.monotonic()
            mc_time = end_time - start_time
            total_time = unfold_time + mc_time
            if deadline and total_time * 1000 > deadline:
                timeout = True
            writer.writerow([i, observation, risk, unfold_time, mc_time, total_time, mdp.nr_states, mdp.nr_transitions, timeout])
            file.flush()
            if terminate_on_deadline and timeout:
                return False
    return True

class ForwardFilteringOptions:
    """
    Container to configure forward filtering.
    """
    def __init__(self, convex_hull_reduction=True, exact_arithmetic=True):
        self.convex_hull_reduction = convex_hull_reduction
        self.exact_arithmetic = exact_arithmetic

    @property
    def method_id(self):
        redstr = "ch" if self.convex_hull_reduction else "nr"
        numstr = "ea" if self.exact_arithmetic else "fl"
        return "ff-{}-{}".format(redstr, numstr)

class UnfoldingOptions:
    """
    Container to configure unfolding.
    """
    def __init__(self, exact_arithmetic=True):
        self.exact_arithmetic = exact_arithmetic

    @property
    def method_id(self):
        numstr = "ea" if self.exact_arithmetic else "fl"
        return "unf-{}".format(numstr)

def monitor(path, risk_property, constants, trace_length, options, verbose=False, simulator_seed=0, promptness_deadline=10000, model_id="no_id_given"):
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
    prism_program = sp.parse_prism_program(path)
    prop = sp.parse_properties_for_prism_program(risk_property, prism_program)[0]
    prism_program, props = sp.preprocess_symbolic_input(prism_program, [prop], constants)
    prop = props[0]
    prism_program = prism_program.as_prism_program()
    raw_formula = prop.raw_formula

    logger.info("Construct MDP representation...")
    model = build_model(prism_program, raw_formula, exact_arithmetic=options.exact_arithmetic)
    if(verbose):
        print(model)
    assert model.has_observation_valuations()
    logger.info("Compute risk per state")
    risk_assessment = analyse_model(model, prop).get_values()

    # The seed can be given as a single value or as a
    if isinstance(simulator_seed, Iterable):
        simulator_seed_range = simulator_seed
    else:
        simulator_seed_range = range(simulator_seed)

    logger.info("Initialize simulator...")
    simulator = sp.simulator.create_simulator(model)
    if use_forward_filtering:
        logger.info("Initialize tracker...")
        tracker = stormpy.pomdp.create_nondeterminstic_belief_tracker(model, promptness_deadline, promptness_deadline)
        tracker.set_risk(risk_assessment)
    if use_unfolding:
        expr_manager = stormpy.ExpressionManager()
        unfolder = stormpy.pomdp.create_observation_trace_unfolder(model, risk_assessment, expr_manager)

    initialize_time = time.monotonic() - start_time
    #stormpy.export_to_drn(model, "model.drn")
    stats_folder = f"stats/{model_id}-{options.method_id}/"
    if not os.path.isdir(stats_folder):
        os.makedirs(stats_folder)
    else:
        raise RuntimeWarning("We are writing to an existing folder.")
    with open(os.path.join(stats_folder,"stats.out"), 'w') as file:
        file.write(f"states={model.nr_states}\n")
        file.write(f"transitions={model.nr_transitions}\n")
        file.write(f"init_time={initialize_time}\n")
        file.write(f"promptness_deadline={promptness_deadline}")

    for seed in tqdm(simulator_seed_range):
        simulator.set_seed(seed)
        logger.info("Restart simulator...")
        observation, _ = simulator.restart()

        stats_file = f"{stats_folder}/stats-{model_id}-{options.method_id}-{seed}.csv"
        if use_forward_filtering:
            filtering(simulator, tracker, trace_length, options.convex_hull_reduction, stats_file, deadline=promptness_deadline, verbose=verbose, observation_valuations=model.observation_valuations)
        if use_unfolding:
            unfolding(simulator, unfolder, trace_length, stats_file, use_ovi=not options.exact_arithmetic, deadline=promptness_deadline)



