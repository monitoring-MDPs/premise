from dataclasses import dataclass
from itertools import product
import json
from typing import Any
import stormpy as sp
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class ModelDescription:
    prism_program_path: Path
    constants: str
    risk_property: str
    target_label: str


default_models = {
    "airportA-7-5-5": ModelDescription(
        Path(__file__).parent / "examples/airportA-7.nm",
        "DMAX=10,PMAX=10",
        'Pmax=? [F "crash"]',
        "crash",
    ),
    "airportA-7-50-30": ModelDescription(
        Path(__file__).parent / "examples/airportA-7.nm",
        "DMAX=50,PMAX=30",
        'Pmax=? [F "crash"]',
        "crash",
    ),
    "airportB-3-50-30": ModelDescription(
        Path(__file__).parent / "examples/airportB-3.nm",
        "DMAX=50,PMAX=30",
        'Pmax=? [F "crash"]',
        "crash",
    ),
    "airportB-7-50-30": ModelDescription(
        Path(__file__).parent / "examples/airportB-7.nm",
        "DMAX=50,PMAX=30",
        'Pmax=? [F "crash"]',
        "crash",
    ),
    "evadeI-15": ModelDescription(
        Path(__file__).parent / "examples/hidden-incentive.nm",
        "N=15",
        'Pmax=? [F<=12 "crash"]',
        "crash",
    ),
    "evadeV-5-3": ModelDescription(
        Path(__file__).parent / "examples/evade-monitoring.nm",
        "N=5,RADIUS=3",
        'Pmax=? [F<=12 "crash"]',
        "crash",
    ),
    "evadeV-6-3": ModelDescription(
        Path(__file__).parent / "examples/evade-monitoring.nm",
        "N=6,RADIUS=3",
        'Pmax=? [F<=12 "crash"]',
        "crash",
    ),
    "refuelA-12-50": ModelDescription(
        Path(__file__).parent / "examples/refuel.nm",
        "N=12,ENERGY=50",
        'Pmax=? [F<=12 "empty"]',
        "empty",
    ),
    "refuelB-12-50": ModelDescription(
        Path(__file__).parent / "examples/refuelB.nm",
        "N=12,ENERGY=50",
        'Pmax=? [F<=12 "empty"]',
        "empty",
    ),
    "SnL-10x10": ModelDescription(
        Path(__file__).parent / "examples/SnL.nm",
        "n=100, l1s=1, l1d=38, l2s=4, l2d=14, l3s=9, l3d=31, l4s=28, l4d=64, l5s=40, l5d=42, l6s=36, l6d=44, l7s=51, l7d=67, l8d=91, l8s=71, l9s=80, l9d=100, l10s=-1, l10d=-1, s1s=98, s1d=76, s2s=95, s2d=75, s3s=93, s3d=73, s4s=87, s4d=24, s5s=64, s5d=60, s6s=62, s6d=19, s7s=55, s7d=53, s8d=11, s8s=49, s9s=47, s9d=26, s10s=16, s10d=6",
        'Pmax=? [F<3 "good" ]',
        "good",
    ),
}


def build_model_and_risk(model_description: ModelDescription, options):
    prism_program = sp.parse_prism_program(str(model_description.prism_program_path))
    prop = sp.parse_properties_for_prism_program(
        model_description.risk_property, prism_program
    )[0]
    prism_program, props = sp.preprocess_symbolic_input(
        prism_program, [prop], model_description.constants
    )
    prop = props[0]
    prism_program = prism_program.as_prism_program()
    raw_formula = prop.raw_formula
    logger.info("Construct MDP representation...")
    model = _build_model(
        prism_program, raw_formula, exact_arithmetic=options.exact_arithmetic
    )
    if options.verbose:
        print(model)
    assert model.has_observation_valuations()
    logger.info("Compute risk per state")
    risk_assessment = _analyse_model(model, prop).get_values()
    return model, risk_assessment


def _build_model(program, formula, exact_arithmetic):
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


def _analyse_model(model, prop):
    """
    Analyse the MDP with respect to the given property. Assume full observability.
    :param model:
    :param prop:
    :return:
    """
    return sp.model_checking(model, prop.raw_formula, force_fully_observable=True)


def build_noaction_model_and_risk(model_description: ModelDescription, options):
    # Build POMDP model
    prism_program = sp.parse_prism_program(str(model_description.prism_program_path))
    prop = sp.parse_properties_for_prism_program(
        model_description.risk_property, prism_program
    )[0]
    prism_program, props = sp.preprocess_symbolic_input(
        prism_program, [prop], model_description.constants
    )
    prop = props[0]
    prism_program = prism_program.as_prism_program()
    raw_formula = prop.raw_formula

    logger.info("Construct MDP representation...")
    pomdp_stormpy = _build_model(
        prism_program, raw_formula, exact_arithmetic=options.exact_arithmetic
    )
    assert pomdp_stormpy.has_observation_valuations()

    # Collapse to only one action per state
    # First build state labeling
    state_labeling = sp.StateLabeling(len(pomdp_stormpy.states))
    for label in pomdp_stormpy.labeling.get_labels():
        state_labeling.add_label(label)

    # Build state valuations
    manager = sp.ExpressionManager()
    state_valuations = sp.storage.StateValuationsBuilder()
    var_order = []
    init_val = json.loads(str(pomdp_stormpy.state_valuations.get_json(0)))
    for var in init_val.keys():
        storm_var = manager.create_integer_variable(var)
        state_valuations.add_variable(storm_var)
        var_order.append(var)

    for state in pomdp_stormpy.states:
        if state.id >= pomdp_stormpy.state_valuations.get_nr_of_states():
            raise ValueError(
                f"State {state.id} is not in the state valuations. This should not happen."
            )
        val = json.loads(str(pomdp_stormpy.state_valuations.get_json(state.id)))
        state_valuations.add_state(state.id, integer_values=[val[v] for v in var_order])

    obs_valuations = sp.storage.StateValuationsBuilder()
    obs_var_order = []
    init_obs_val = json.loads(str(pomdp_stormpy.observation_valuations.get_json(0)))
    for var in init_obs_val.keys():
        try:
            storm_var = manager.get_variable(var)
        except:
            storm_var = manager.create_integer_variable(var)
        obs_valuations.add_variable(storm_var)
        obs_var_order.append(var)

    for obs in range(pomdp_stormpy.observation_valuations.get_nr_of_states()):
        if obs >= pomdp_stormpy.observation_valuations.get_nr_of_states():
            raise ValueError(
                f"Observation {obs} is not in the observation valuations. This should not happen."
            )
        val = json.loads(str(pomdp_stormpy.observation_valuations.get_json(obs)))
        obs_valuations.add_state(obs, integer_values=[val[v] for v in obs_var_order])

    # Create transition matrix
    if options.exact_arithmetic:
        builder = sp.ExactSparseMatrixBuilder(0, 0, 0, False, False)
    else:
        builder = sp.SparseMatrixBuilder(0, 0, 0, False, False)
    for s in pomdp_stormpy.states:
        # Set labels
        for label in s.labels:
            state_labeling.add_label_to_state(label, s.id)

        # Set transition
        amount_of_actions = len(s.actions)
        new_row_dict: dict[int, Any] = {}
        for action in s.actions:
            for transition in action.transitions:
                dest_s = transition.column
                if dest_s in new_row_dict:
                    new_row_dict[dest_s] += transition.value() / amount_of_actions
                else:
                    new_row_dict[dest_s] = transition.value() / amount_of_actions

        for new_dest_s, value in sorted(new_row_dict.items()):
            builder.add_next_value(s.id, new_dest_s, value)

    matrix = builder.build(overridden_column_count=len(pomdp_stormpy.states))

    if options.exact_arithmetic:
        components = sp.SparseExactModelComponents(matrix, state_labeling)
        components.state_valuations = state_valuations.build()
        components.observation_valuations = obs_valuations.build()
        components.observability_classes = pomdp_stormpy.observations
        model = sp.SparseExactPomdp(components)
    else:
        components = sp.SparseModelComponents(matrix, state_labeling)
        components.state_valuations = state_valuations.build()
        components.observation_valuations = obs_valuations.build()
        components.observability_classes = pomdp_stormpy.observations
        model = sp.SparsePomdp(components)

    risk_assessment = _analyse_model(model, prop).get_values()
    return model, risk_assessment


def build_state_and_transition_list(
    model, target_label: str, all_transitions=False, add_label_to_state=False
):
    """
    Build a list of states and a list of transitions from the model
    :param model: The model to extract the states and transitions from
    :return: A tuple with the list of states and the list of transitions
    """
    states_map = {}
    for state in model.states:
        if add_label_to_state:
            states_map[state.id] = (
                (state.id, model.state_valuations.get_string(state.id)),
                (
                    model.get_observation(state.id),
                    model.observation_valuations.get_string(
                        model.get_observation(state.id)
                    ),
                ),
                model.labeling.has_state_label(target_label, state.id),
            )
        else:
            states_map[state.id] = (
                state.id,
                model.get_observation(state.id),
                model.labeling.has_state_label(target_label, state.id),
            )

    initial_states = []
    for state_id in model.initial_states:
        initial_states.append(states_map[state_id])

    if all_transitions:
        return (
            list(states_map.values()),
            list(product(states_map.values(), states_map.values())),
            list(states_map.values()),
        )
    else:
        transitions = set()
        for state in model.states:
            for action in state.actions:
                for transition in action.transitions:
                    transitions.add(
                        (states_map[state.id], states_map[transition.column])
                    )

        return list(states_map.values()), list(transitions), initial_states
