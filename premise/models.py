from dataclasses import dataclass
import stormpy as sp
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# The example models live in the top-level `examples` folder of the repository.
EXAMPLES_DIR = Path(__file__).resolve().parent.parent / "examples"


@dataclass
class ModelDescription:
    prism_program_path: str
    constants: str
    risk_property: str


default_models =  {
    "airportA-7-50-30" : ModelDescription(EXAMPLES_DIR / "airportA-7.nm", "DMAX=50,PMAX=30", "Pmax=? [F \"crash\"]"),
    "airportB-3-50-30": ModelDescription(EXAMPLES_DIR / "airportB-3.nm", "DMAX=50,PMAX=30", "Pmax=? [F \"crash\"]"),
    "airportB-7-50-30": ModelDescription( EXAMPLES_DIR / "airportB-7.nm", "DMAX=50,PMAX=30", "Pmax=? [F \"crash\"]"),
    "evadeI-15": ModelDescription(EXAMPLES_DIR / "hidden-incentive.nm", "N=15", "Pmax=? [F<=12 \"crash\"]"),
    "evadeV-5-3": ModelDescription(EXAMPLES_DIR / "evade-monitoring.nm", "N=5,RADIUS=3", "Pmax=? [F<=12 \"crash\"]"),
    "evadeV-6-3": ModelDescription(EXAMPLES_DIR / "evade-monitoring.nm", "N=6,RADIUS=3", "Pmax=? [F<=12 \"crash\"]"),
    "refuelA-12-50": ModelDescription(EXAMPLES_DIR / "refuel.nm", "N=12,ENERGY=50", "Pmax=? [F<=12 \"empty\"]"),
    "refuelB-12-50": ModelDescription(EXAMPLES_DIR / "refuelB.nm", "N=12,ENERGY=50", "Pmax=? [F<=12 \"empty\"]")
}



def build_model_and_risk(model_description : ModelDescription, options):
    prism_program = sp.parse_prism_program(str(model_description.prism_program_path))
    prop = sp.parse_properties(model_description.risk_property, prism_program)[0]
    prism_program, props = sp.instantiate_constants(prism_program, model_description.constants, [prop])
    prop = props[0]
    logger.info("Construct MDP representation...")
    model = _build_model(prism_program, [prop], exact_arithmetic=options.exact_arithmetic)
    if (options.verbose):
        print(model)
    assert model.has_observation_valuations()
    logger.info("Compute risk per state")
    risk_assessment = _analyse_model(model, prop).get_value_vector()
    return model, risk_assessment


def _build_model(program, props, exact_arithmetic):
    """
    Takes a model and the properties that should be checkable on it (e.g., a property that describes the bad event).

    :param program: PRISM program
    :param props: PCTL specifications whose labels are built, e.g. a specification for a bad event
    :param exact_arithmetic: Flag for using exact arithmetic.
    :return: A sparse MDP
    """
    options = sp.BuilderOptions(props)
    options.set_build_state_valuations()
    options.set_build_observation_valuations()
    options.set_build_choice_labels()
    options.set_build_all_labels()
    logger.debug("Start building the MDP")
    value_type = sp.Rational if exact_arithmetic else float
    return sp.build_model[value_type](program, options=options)



def _analyse_model(model, prop):
    """
    Analyse the MDP with respect to the given property. Assume full observability.
    :param model:
    :param prop:
    :return:
    """
    return sp.model_checking(model, prop, force_fully_observable=True)
