import copy
from pathlib import Path
import sys
from multiprocessing import Pool, TimeoutError
import traceback
from typing import Optional

import numpy as np

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


def run_with_timeout(func, args, timeout, prev_stats_path: Optional[Path] = None):
    if prev_stats_path:
        stat_path = prev_stats_path / Path(args[0].dump_stats).name
        if stat_path.exists():
            logger.info(f"Skipping experiment {args[0]}, already exists: {stat_path}")
            try:
                stats = np.load(stat_path, allow_pickle=True).item()
            except AttributeError:
                try:
                    stats = np.load(stat_path, allow_pickle=True)
                except Exception as e:
                    stats = {}
                    logger.error(f"Failed to load existing stats from {stat_path}: {e}")

            if "unfinished" in stats:
                logger.info(f"Found unfinished stats, redoing experiment {args[0]}.")
            else:
                logger.info(f"Using existing stats for {args[0]}.")
                return stats

    with Pool(processes=1) as pool:
        async_result = pool.apply_async(func, args)
        try:
            result = async_result.get(timeout)
        except Exception as e:
            pool.terminate()
            pool.join()
            raise e
        return result


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

    if sys.argv[2] != "-":
        prev_stats_path = Path(sys.argv[2])
        if not prev_stats_path.exists():
            logger.error(f"Previous stats file {prev_stats_path} does not exist.")
            sys.exit(1)
    else:
        prev_stats_path = None

    if len(args) != 3:
        print(
            "Usage: python run.py comp_methods <args refinement> :: <args regression> :: <args mle> :: <args conformal prediction>",
            args,
            sys.argv,
            file=sys.stderr,
        )
        sys.exit(2)

    # NORMAL REFINEMENT without splitting
    ref_parser = ref_args_parser()

    ref_args = ref_parser.parse_args(args[0])
    try:
        ref_stats = run_with_timeout(ref_main, (ref_args,), timeout, prev_stats_path)
        ref_trans_count = ref_stats["transition_count"]
    except TimeoutError:
        logger.warning("Refinement timed out.")
        if Path(ref_args.dump_stats).exists():
            data = np.load(ref_args.dump_stats, allow_pickle=True).item()
            ref_trans_count = data.get("transition_count", 0)
            logger.info(
                f"Found unfinished stats, using last transition count: {ref_trans_count}"
            )
        else:
            np.save(ref_args.dump_stats, {"timed_out": True})  # type: ignore
            ref_trans_count = 0
    except Exception as e:
        logger.error(f"Error in refinement: {traceback.format_exc()}")
        ref_trans_count = 0

    # NORMAL REFINEMENT with splitting
    ref_split_args = copy.deepcopy(ref_args)
    ref_split_args.model_path = ref_args.model_path.replace("ref", "refsplit")
    ref_split_args.dump_stats = ref_args.dump_stats.replace("ref", "refsplit")
    ref_split_args.use_splitting = True
    try:
        ref_split_stats = run_with_timeout(
            ref_main, (ref_split_args,), timeout, prev_stats_path
        )
        ref_split_trans_count = ref_split_stats["transition_count"]
    except TimeoutError:
        logger.warning("Refinement with splitting timed out.")
        if Path(ref_split_args.dump_stats).exists():
            data = np.load(ref_split_args.dump_stats, allow_pickle=True).item()
            ref_split_trans_count = data.get("transition_count", 0)
            logger.info(
                f"Found unfinished stats, using last transition count: {ref_split_trans_count}"
            )
        else:
            np.save(ref_split_args.dump_stats, {"timed_out": True})  # type: ignore
            ref_split_trans_count = 0
    except Exception as e:
        logger.error(f"Error in refinement with splitting: {traceback.format_exc()}")
        ref_split_trans_count = 0

    transition_count = max(ref_trans_count, ref_split_trans_count)
    if transition_count == 0:
        logger.warning("No transitions were made in refinement, stopping experiment.")
        sys.exit(3)

    logger.info(f"Transitions from refinement: {transition_count}")

    no_ref = copy.deepcopy(ref_args)
    no_ref.model_path = ref_args.model_path.replace("ref", "noref")
    no_ref.dump_stats = ref_args.dump_stats.replace("ref", "noref")
    no_ref.stopping_samples = transition_count
    no_ref.stopping_criteria = "samples"
    try:
        run_with_timeout(ref_main, (no_ref,), timeout, prev_stats_path)
    except TimeoutError:
        np.save(no_ref.dump_stats, {"timed_out": True})  # type: ignore
        logger.warning("No-refinement timed out, continue to regression.")
    except Exception as e:
        logger.error(f"Error in no-refinement: {traceback.format_exc()}")
        logger.info("No-refinement failed, continuing to regression.")

    # REGRESSION
    reg_parser = reg_argsparser()
    reg_args = reg_parser.parse_args(args[1])

    model_def = default_models[ref_args.mc]
    horizon = model_def.horizon
    initial_amount = model_def.initial_amount

    reg_args.amount = transition_count // (horizon + initial_amount)

    try:
        run_with_timeout(reg_main, (reg_args,), timeout, prev_stats_path)
    except TimeoutError:
        np.save(reg_main.dump_stats, {"timed_out": True})  # type: ignore
        logger.warning("Regression timed out.")
    except Exception as e:
        logger.error(f"Error in regression: {traceback.format_exc()}")
        logger.info("Regression failed, continuing to MLE.")

    # MAXIMUM LIKELIHOOD ESTIMATION
    mle_parser = mle_args_parser()
    mle_args = mle_parser.parse_args(args[2])
    mle_args.samples = transition_count // (horizon + initial_amount)

    try:
        run_with_timeout(mle_learning_main, (mle_args,), timeout, prev_stats_path)
    except TimeoutError:
        np.save(mle_args.dump_stats, {"timed_out": True})  # type: ignore
        logger.warning("MLE timed out.")
    except Exception as e:
        logger.error(f"Error in MLE: {traceback.format_exc()}")

        logger.info("MLE failed, continuing to Conformal Prediction.")

    # CONFORMAL PREDICTION
    # conformal_parser = conformal_prediction_argsparser()
    # conformal_args = conformal_parser.parse_args(args[3])
    # conformal_args.amount = transition_count // (horizon + initial_amount)

    # try:
    #     run_with_timeout(conformal_prediction_main, (conformal_args,), timeout)
    # except TimeoutError:
    #     logger.warning("Conformal Prediction timed out.")
