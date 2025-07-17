import copy
import resource
import sys
from multiprocessing import Pool
from pathlib import Path
import pickle
import re
import matplotlib.pyplot as plt
import numpy as np
from sympy import use

from premise.interval.maximum_likelihood import mle_args_parser, mle_learning_main
from premise.interval.utils import setup_logging
from premise.interval.model_free.regression_model import reg_argsparser, reg_main
from premise.interval.refinement import ref_args_parser, ref_main
from premise.interval.utils import logger
from premise.interval.conformal_prediction.conformal_prediction import (
    conformal_prediction_argsparser,
    conformal_prediction_main,
)
from premise.models import default_models


def split_args(args, delim):
    res = [[]]
    for a in args:
        if a == delim:
            res.append([])
        else:
            res[-1].append(a)
    return res



def run_with_timeout(func, args, timeout):

    with Pool(processes=1) as pool:
        async_result = pool.apply_async(func, args)
        try:
            result = async_result.get(timeout)
        except Exception as e:
            pool.terminate()
            pool.join()
            if isinstance(e, TimeoutError):
                logger.warning("Timeout occurred")
                raise TimeoutError("Timeout occurred")
            raise e
        return result


    
def get_transition_count(stats_path, model_def):
    transition_stats = []
    path = Path(stats_path) 
    if path.is_dir():
        paths = path.iterdir()
    else:
        paths = [path]

    for stat_path in paths:
        if model_def in str(stat_path):
            if '-ref-' in str(stat_path) or '-refsplit-' in str(stat_path):
                    try:
                        data = np.load(stat_path, allow_pickle=True).item()
                    except AttributeError:
                        data = pickle.load(stat_path.open("rb"))
                    try:
                        transition_stats.append(np.sum(data["transitions_learned"]))
                    except Exception as e:
                        print(f"Error processing {stat_path}: {e} ({data})")

    return max(transition_stats)

           

if __name__ == "__main__":
    args = split_args(sys.argv[3:], "::")

    setup_logging()

    logger.info(f"Running with args: {args}")

    # Calculate timeout
    if sys.argv[1][-1] == "h":
        timeout = int(sys.argv[1][:-1]) * 60 * 60
    elif sys.argv[1][-1] == "m":
        timeout = int(sys.argv[1][:-1]) * 60
    elif sys.argv[1][-1] == "s":
        timeout = int(sys.argv[1][:-1])
    else:
        logger.error(
            "Invalid timeout format. Use <number>[h|m|s] (e.g., 1h, 30m, 45s)."
        )
        sys.exit(2)

    if len(args) != 1:
            print(
                "Usage: python run_conformal_prediction.py ",
                args,
                sys.argv,
                file=sys.stderr,
            )
            sys.exit(2)

    conformal_parser = conformal_prediction_argsparser()
    conformal_args = conformal_parser.parse_args(args[0])

    model_def = default_models[conformal_args.mc]
    horizon = model_def.horizon
    initial_amount = model_def.initial_amount

    transition_count = get_transition_count('/workspaces/premise/out/stats/2025-07-10_07-55-18', model_def)

    conformal_args.amount = transition_count // (horizon + initial_amount)

    try:
        run_with_timeout(conformal_prediction_main, (conformal_args,), timeout)
    except TimeoutError:
        logger.warning("Conformal Prediction timed out.")


    