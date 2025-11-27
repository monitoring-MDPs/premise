from argparse import ArgumentParser, Namespace

import numpy as np

from premise.system import (
    CarlaSimSystemUnderObservation,
    CoarseMCSystemUnderObservation,
    MCSystemUnderObservation,
    SystemUnderObservation,
    CarlaPreSampledSystemUnderObservation,
    # ACASSystemUnderObservation,
)
from premise.models import default_models
from premise.interval.utils import logger


def build_suo_args_parser(parser: ArgumentParser, required: bool = True):
    group = parser.add_argument_group("System Under Observation")
    model_group = group.add_mutually_exclusive_group(required=required)
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
    model_group.add_argument(
        "-acas",
        "--acas",
        type=float,
        help="Use the ACAS model with specified coarseness factor, e.g., 2 is twice as coarse as 1.",
    )


def build_imc_loading_args_parser(
    parser: ArgumentParser, required: bool = True, option_prefix=""
):
    group = parser.add_argument_group("IMC loading")
    group.add_argument(
        f"-{option_prefix}t",
        f"--{option_prefix}trans-path",
        required=required,
        type=str,
        default=None,
        help="Path to the transition dictionary",
    )
    group.add_argument(
        f"-{option_prefix}i",
        f"--{option_prefix}init-path",
        required=required,
        type=str,
        default=None,
        help="Path to the initial state dictionary",
    )


def build_suo(
    args: Namespace,
) -> tuple[
    SystemUnderObservation,
    int,
    int,
]:
    horizon = None
    initial_amount = None

    logger.info(
        f"Building System Under Observation {args.mc or args.sam or args.sim or args.acas}"
    )

    if args.mc:
        model_def = default_models[args.mc]
        logger.info(f"Using model definition: {model_def}")
        model_def.risk_property = f'Pmax=? [F<={vars(args).get("horizon", model_def.horizon) or model_def.horizon} "{model_def.target_label}" ]'
        horizon = model_def.horizon
        initial_amount = model_def.initial_amount

        if args.sys_vars is not None:
            suo: SystemUnderObservation = CoarseMCSystemUnderObservation(
                model_def, args.mc, args.sys_vars
            )
            if args.verbose > 1:
                for s, c in suo.state_coarse_map.items():
                    logger.info(
                        f"{suo._model.state_valuations.get_string(s)}: {c} [{float(suo.risk[s])}]"
                    )
        else:
            suo: SystemUnderObservation = MCSystemUnderObservation(model_def, args.mc)
            if args.verbose > 1:
                for i, r in enumerate(suo.get_risk()):
                    logger.info(
                        f"{suo._model.state_valuations.get_string(i)}: {float(r)}"
                    )
    elif args.sam:
        suo: SystemUnderObservation = CarlaPreSampledSystemUnderObservation(args.sam)
    elif args.sim:
        suo: SystemUnderObservation = CarlaSimSystemUnderObservation(args.sim)
    # elif args.acas:
    #     suo: SystemUnderObservation = ACASSystemUnderObservation(
    #         args.acas, vars(args).get("horizon", 1)
    #     )
    else:
        raise ValueError("No model specified")

    if horizon is None:
        horizon = args.get("horizon", 1)
    if initial_amount is None:
        initial_amount = args.get("initial_amount", 0)

    return suo, initial_amount, horizon


def load_imc(args: Namespace):
    interval = np.load(args.trans_path, allow_pickle=True)[()]
    initial_interval = np.load(args.init_path, allow_pickle=True)[()]
    return interval, initial_interval
