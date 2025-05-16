from argparse import ArgumentParser, Namespace

import numpy as np

from premise.system import (
    CarlaSimSystemUnderObservation,
    CoarseMCSystemUnderObservation,
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
    group.add_argument(
        "-sv",
        "--sys-vars",
        nargs="+",
        type=str,
        default=None,
        help="System variables to be used",
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


def build_imc_loading_args_parser(parser: ArgumentParser, required: bool = True):
    group = parser.add_argument_group("IMC loading")
    group.add_argument(
        "-t",
        "--trans_path",
        required=required,
        type=str,
        default=None,
        help="Path to the transition dictionary",
    )
    group.add_argument(
        "-i",
        "--init_path",
        required=required,
        type=str,
        default=None,
        help="Path to the initial state dictionary",
    )


def build_suo(args: Namespace):
    if args.mc:
        model_def = default_models[args.mc]
        model_def.risk_property = (
            f'Pmax=? [F<={vars(args).get("horizon", 1)} "{model_def.target_label}" ]'
        )
        if args.sys_vars is not None:
            suo: SystemUnderObservation = CoarseMCSystemUnderObservation(
                model_def, args.mc, args.sys_vars
            )
            if args.verbose > 1:
                for s, c in suo.state_coarse_map.items():
                    print(
                        f"{suo._model.state_valuations.get_string(s)}: {c} [{float(suo.risk[s])}]"
                    )
        else:
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
