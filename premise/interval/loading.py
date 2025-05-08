from argparse import ArgumentParser, Namespace

import numpy as np

from premise.system import (
    CarlaSimSystemUnderObservation,
    MCSystemUnderObservation,
    SystemUnderObservation,
    CarlaPreSampledSystemUnderObservation,
)
from premise.models import default_models


def build_suo_args_parser(parser: ArgumentParser):
    group = parser.add_argument_group("System Under Observation")
    model_group = group.add_mutually_exclusive_group(required=True)
    model_group.add_argument(
        "-mc", "--mc", type=str, help="Use the premise model with the given name"
    )
    model_group.add_argument(
        "-sam",
        "--sam",
        nargs="+",
        type=str,
        help="Use the simulation model with the given name",
    )
    model_group.add_argument(
        "-sim",
        "--sim",
        type=str,
        help="Use the simulation model with the given name",
    )


def build_imc_loading_args_parser(parser: ArgumentParser):
    group = parser.add_argument_group("IMC loading")
    group.add_argument(
        "-t",
        "--trans_path",
        required=True,
        type=str,
        help="Path to the transition dictionary",
    )
    group.add_argument(
        "-i",
        "--init_path",
        required=True,
        type=str,
        help="Path to the initial state dictionary",
    )


def build_suo(args: Namespace):
    if args.mc:
        model_def = default_models[args.mc]
        model_def.risk_property = (
            f'Pmax=? [F<={args.horizon} "{model_def.target_label}" ]'
        )
        suo: SystemUnderObservation = MCSystemUnderObservation(model_def, args.mc)

        if args.verbose > 1:
            for i, r in enumerate(suo.get_risk()):
                print(f"{suo._model.state_valuations.get_string(i)}: {float(r)}")
    elif args.sam:
        suo: SystemUnderObservation = CarlaPreSampledSystemUnderObservation(args.sam)
    elif args.sim:
        suo: SystemUnderObservation = CarlaSimSystemUnderObservation(args.sim)
    else:
        raise ValueError("No model specified")
    return suo


def load_imc(args: Namespace):
    interval = np.load(args.trans_path, allow_pickle=True)[()]
    initial_interval = np.load(args.init_path, allow_pickle=True)[()]
    return interval, initial_interval
