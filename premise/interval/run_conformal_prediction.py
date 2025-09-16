import copy
import sys
from multiprocessing import Pool, set_start_method
from pathlib import Path
import pickle
import numpy as np

from premise.interval.utils import setup_logging
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
            raise e
        return result


def get_transition_count(stats_path: str, model_def_name: str):
    transition_stats = []
    high_st_transition_stats = []

    path = Path(stats_path)
    if path.is_dir():
        paths = path.iterdir()
    else:
        paths = [path]

    if model_def_name == 'SnL-10x10': 
        model_name = 'SnL'
    else: 
        model_name = model_def_name



    for stat_path in paths:
        if model_name in str(stat_path):

            if "-ref-" in str(stat_path) or "-refsplit-" in str(stat_path):
                try:
                    data = np.load(stat_path, allow_pickle=True).item()
                except AttributeError:
                    data = pickle.load(stat_path.open("rb"))
                try:
                    if "high-st" in str(stat_path):
                        high_st_transition_stats.append(data["transition_count"])
                    else:
                        transition_stats.append(data["transition_count"])
                except Exception as e:
                    print(f"Error processing {stat_path}: {e} ({data})")

    if len(transition_stats) == 0:
        logger.error(
            f"No transition stats found for model {model_def_name} in {stats_path} for regular st."
        )
        sys.exit(1)
    if len(high_st_transition_stats) == 0:
        logger.error(
            f"No transition stats found for model {model_def_name} in {stats_path} for high st."
        )
        sys.exit(1)

    return max(transition_stats), max(high_st_transition_stats)
    

if __name__ == "__main__":
    args = split_args(sys.argv[3:], "::")

    setup_logging()

    set_start_method("spawn")

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

    transition_count, high_st_transition_count = get_transition_count(
    sys.argv[2], conformal_args.mc)

    high_st_args = copy.deepcopy(conformal_args)

    conformal_args.amount = transition_count // (horizon + initial_amount)

    high_st_args.amount = high_st_transition_count // (horizon + initial_amount)
    
    high_st_args.high_st = True

    try:
        run_with_timeout(conformal_prediction_main, (high_st_args,), timeout)
    except TimeoutError:
        logger.warning("High st Conformal Prediction timed out.")
    except Exception as e:
        logger.error(f"Error in Conformal Prediction: {e}")
        logger.info("Conformal Prediction failed, exiting.")

    try:
        run_with_timeout(conformal_prediction_main, (conformal_args,), timeout)
    except TimeoutError:
        logger.warning("Conformal Prediction timed out.")
    except Exception as e:
        logger.error(f"Error in Conformal Prediction: {e}")
        logger.info("Conformal Prediction failed, exiting.")

# python -m premise.interval.run_conformal_prediction 60m /workspaces/premise/out/stats/2025-09-16_07-48-07 -mc SnL-10x10 --dump-model /workspaces/premise/out/models/2025-09-16_07-48-07 --dump-stats /workspaces/premise/out/stats/2025-09-16_07-48-07 --no-target
