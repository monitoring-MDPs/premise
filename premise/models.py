from dataclasses import dataclass
import stormpy as sp
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class ModelDescription:
    prism_program_path: str
    constants: str
    risk_property: str


default_models =  {
    "airportA-7-50-30" : ModelDescription(Path(__file__).parent  / "examples/airportA-7.nm", "DMAX=50,PMAX=30", "Pmax=? [F \"crash\"]"),
    "airportB-3-50-30": ModelDescription(Path(__file__).parent / "examples/airportB-3.nm", "DMAX=50,PMAX=30", "Pmax=? [F \"crash\"]"),
    "airportB-7-50-30": ModelDescription( Path(__file__).parent / "examples/airportB-7.nm", "DMAX=50,PMAX=30", "Pmax=? [F \"crash\"]"),
    "evadeI-15": ModelDescription(Path(__file__).parent / "examples/hidden-incentive.nm", "N=15", "Pmax=? [F<=12 \"crash\"]"),
    "evadeV-5-3": ModelDescription(Path(__file__).parent / "examples/evade-monitoring.nm", "N=5,RADIUS=3", "Pmax=? [F<=12 \"crash\"]"),
    "evadeV-6-3": ModelDescription(Path(__file__).parent / "examples/evade-monitoring.nm", "N=6,RADIUS=3", "Pmax=? [F<=12 \"crash\"]"),
    "refuelA-12-50": ModelDescription(Path(__file__).parent / "examples/refuel.nm", "N=12,ENERGY=50", "Pmax=? [F<=12 \"empty\"]"),
    "refuelB-12-50": ModelDescription(Path(__file__).parent / "examples/refuelB.nm", "N=12,ENERGY=50", "Pmax=? [F<=12 \"empty\"]")
}



def build_model_and_risk(model_description : ModelDescription, options):
    prism_program = sp.parse_prism_program(str(model_description.prism_program_path))
    prop = sp.parse_properties_for_prism_program(model_description.risk_property, prism_program)[0]
    prism_program, props = sp.preprocess_symbolic_input(prism_program, [prop], model_description.constants)
    prop = props[0]
    prism_program = prism_program.as_prism_program()
    raw_formula = prop.raw_formula
    logger.info("Construct MDP representation...")
    model = _build_model(prism_program, raw_formula, exact_arithmetic=options.exact_arithmetic)
    if (options.verbose):
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
