from dataclasses import dataclass
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
        "n=100, l1s=1, l1d=38, l2s=4, l2d=14, l3s=9, l3d=31, l4s=28, l4d=64, l5s=40, l5d=42, l6s=36, l6d=44, l7s=51, l7d=67, l8d=91, l8s=71, l9s=80, l9d=100, l10s=-1, l10d=-1, s1s=98, s1d=76, s2s=95, s2d=75, s3s=93, s3d=73, s4s=87, s4d=24, s5s=64, s5d=60, s6s=62, s6d=19, s7s=55, s7d=53, s8d=11, s8s=49, s9s=47, s9d=26, s10s=6, s10d=6",
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


def build_state_and_transition_list(model, target_label: str, add_label_to_state=False):
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

    transitions = set()
    for state in model.states:
        for action in state.actions:
            for transition in action.transitions:
                transitions.add((states_map[state.id], states_map[transition.column]))

    return list(states_map.values()), list(transitions)
