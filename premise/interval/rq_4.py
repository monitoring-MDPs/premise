import argparse
import glob
import numpy as np
from tqdm import tqdm
import os
import pickle


from premise.interval.model_free.regression_model import (
    prep_traces_onehot_encoder,
)
from premise.interval.utils import setup_logging
from premise.interval.conformal_prediction.train_stoch_seq_nsc import *
from premise.interval.conformal_prediction.train_seq_se import *
from premise.interval.conformal_prediction.train_seq_nsc import *
from premise.interval.conformal_prediction.CP_Classification import *
from premise.interval.conformal_prediction.CP_Regression import *
from premise.interval.conformal_prediction.SeqDataset import *
import torch
from torch.autograd import Variable
import premise.interval.conformal_prediction.utility_functions as utils
import numpy as np
import argparse
from premise.interval.conformal_prediction.InvertedPendulum import *
from premise.interval.conformal_prediction.MC_model import *
import torch.nn.functional

from premise.interval.loading import (
    build_suo,
    build_suo_args_parser,
    load_imc,
)
from premise.interval.conformence import test_monitor
from premise.interval.interval import (
    Trace,
    create_monitor,
)


def stats_true(horizon, initial_amount, testing_samples, suo):

    mon_create_time = time.monotonic()
    mon = suo.create_target_monitor()
    mon_create_time = time.monotonic() - mon_create_time

    sample_times = []
    for trace in tqdm(testing_samples):
        start_time = time.monotonic()
        sub_trace: Trace = trace[:initial_amount]

        test_monitor(
            mon,
            [sub_trace],
            with_tqdm=False,
        )[sub_trace]

        sample_times.append(time.monotonic() - start_time)

    return {"monitor_creation_time": mon_create_time, "sample_times": sample_times}


def aggregted_alarms(testing_samples):
    alarms = []

    for trace in tqdm(testing_samples):
        alarms.append(any([s[2] for s in trace]))

    alarms = np.array(alarms).astype(int)

    return alarms


def aggregated_stats_imc(
    high_st,
    coarse,
    method,
    mc,
    stats_path,
    initial_amount,
    horizon,
    args,
    testing_samples,
):
    times_stats = {}

    for x in range(1, 11):
        print(f"Experiment number {x}")
        try:
            if coarse:
                if high_st:
                    if method == "noref":
                        if args.mc == "SnLw-10x10":
                            statistics = np.load(
                                f"{stats_path}/high-st-SnL-coarse_norefinement-stats-{x}.npy",
                                allow_pickle=True,
                            )
                        elif args.mc == "evadeV-6-3-coarse":
                            statistics = np.load(
                                f"{stats_path}/high-st-evadeV-6-3-coarse_norefinement-stats-{x}.npy",
                                allow_pickle=True,
                            )
                        else:
                            statistics = np.load(
                                f"{stats_path}/high-st-{mc}-coarse_norefinement-stats-{x}.npy",
                                allow_pickle=True,
                            )
                    elif method == "ref":
                        if args.mc == "SnLw-10x10":
                            statistics = np.load(
                                f"{stats_path}/high-st-SnL-coarse_refinement-stats-{x}.npy",
                                allow_pickle=True,
                            )
                        elif args.mc == "evadeV-6-3-coarse":
                            statistics = np.load(
                                f"{stats_path}/high-st-evadeV-6-3-coarse_refinement-stats-{x}.npy",
                                allow_pickle=True,
                            )
                        else:
                            statistics = np.load(
                                f"{stats_path}/high-st-{mc}-coarse_refinement-stats-{x}.npy",
                                allow_pickle=True,
                            )
                    elif method == "refsplit":
                        if args.mc == "SnLw-10x10":
                            statistics = np.load(
                                f"{stats_path}/high-st-SnL-coarse_refsplitinement-stats-{x}.npy",
                                allow_pickle=True,
                            )
                        elif args.mc == "evadeV-6-3-coarse":
                            statistics = np.load(
                                f"{stats_path}/high-st-evadeV-6-3-coarse_refsplitinement-stats-{x}.npy",
                                allow_pickle=True,
                            )
                        else:
                            statistics = np.load(
                                f"{stats_path}/high-st-{mc}-coarse_refsplitinement-stats-{x}.npy",
                                allow_pickle=True,
                            )
                else:
                    if method == "noref":
                        if args.mc == " SnLw-10x10":
                            statistics = np.load(
                                f"{stats_path}/SnL-coarse_norefinement-stats-{x}.npy",
                                allow_pickle=True,
                            )
                        elif args.mc == "evadeV-6-3-coarse":
                            statistics = np.load(
                                f"{stats_path}/evadeV-6-3-coarse_norefinement-stats-{x}.npy",
                                allow_pickle=True,
                            )
                        else:
                            statistics = np.load(
                                f"{stats_path}/{mc}-coarse_norefinement-stats-{x}.npy",
                                allow_pickle=True,
                            )
                    elif method == "ref":
                        if args.mc == " SnLw-10x10":
                            statistics = np.load(
                                f"{stats_path}/SnL-coarse_refinement-stats-{x}.npy",
                                allow_pickle=True,
                            )
                        elif args.mc == "evadeV-6-3-coarse":
                            statistics = np.load(
                                f"{stats_path}/evadeV-6-3-coarse_refinement-stats-{x}.npy",
                                allow_pickle=True,
                            )
                        else:
                            statistics = np.load(
                                f"{stats_path}/{mc}-coarse_refinement-stats-{x}.npy",
                                allow_pickle=True,
                            )
                    elif method == "refsplit":
                        if args.mc == " SnLw-10x10":
                            statistics = np.load(
                                f"{stats_path}/SnL-coarse_refsplitinement-stats-{x}.npy",
                                allow_pickle=True,
                            )
                        elif args.mc == "evadeV-6-3-coarse":
                            statistics = np.load(
                                f"{stats_path}/evadeV-6-3-coarse_refsplitinement-stats-{x}.npy",
                                allow_pickle=True,
                            )
                        else:
                            statistics = np.load(
                                f"{stats_path}/{mc}-coarse_refsplitinement-stats-{x}.npy",
                                allow_pickle=True,
                            )
            else:
                if high_st:
                    if method == "noref":
                        statistics = np.load(
                            f"{stats_path}/high-st-{mc}-comp-noref-stats-{x}.npy",
                            allow_pickle=True,
                        )
                    elif method == "ref":
                        statistics = np.load(
                            f"{stats_path}/high-st-{mc}-comp-ref-stats-{x}.npy",
                            allow_pickle=True,
                        )
                    elif method == "refsplit":
                        statistics = np.load(
                            f"{stats_path}/high-st-{mc}-comp-refsplit-stats-{x}.npy",
                            allow_pickle=True,
                        )
                else:
                    if method == "noref":
                        statistics = np.load(
                            f"{stats_path}/{mc}-comp-noref-stats-{x}.npy",
                            allow_pickle=True,
                        )
                    elif method == "ref":
                        statistics = np.load(
                            f"{stats_path}/{mc}-comp-ref-stats-{x}.npy",
                            allow_pickle=True,
                        )
                    elif method == "refsplit":
                        statistics = np.load(
                            f"{stats_path}/{mc}-comp-refsplit-stats-{x}.npy",
                            allow_pickle=True,
                        )
        except FileNotFoundError:
            print(f"Statistics file for {x} not found, skipping.")
            logger.warning(f"Statistics file for {x} not found, skipping.")
            continue

        obj = statistics.item()

        model_path = obj["args"]["model_path"]

        args.init_path = f"{model_path}-initial_interval.npy"
        args.trans_path = f"{model_path}-interval.npy"

        transition_intervals, initial_distribution = load_imc(args)

        t = time.monotonic()

        # Build the premise monitor on the learned model
        mon, mon_comps = create_monitor(
            transition_intervals,
            initial_distribution,
            "min",
            True,
            horizon,
            None,
        )
        monitor_created_time = time.monotonic() - t

        sample_times = []
        for t in testing_samples:
            sub_trace: Trace = t[:initial_amount]
            start_time = time.monotonic()
            test_monitor(
                mon,
                [sub_trace],
                obs_func=lambda x: mon_comps.observation_map[x],
                skip_initial=True,
                with_tqdm=False,
            )[sub_trace]
            sample_times.append(time.monotonic() - start_time)

        times_stats[x] = {
            "monitor_creation_time": monitor_created_time,
            "sample_times": sample_times,
        }

    return times_stats


def aggregated_stats_regression(
    high_st,
    coarse,
    mc,
    model_path,
    stats_path,
    testing_samples,
    horizon,
    initial_amount,
):

    time_stats = {}

    for x in range(1, 11):
        try:
            if coarse:
                if high_st:
                    if args.mc == "SnLw-10x10":
                        statistics = np.load(
                            f"{stats_path}/high-st-SnL-coarse-comp-reg-stats-{x}.npy",
                            allow_pickle=True,
                        )
                    elif args.mc == "evadeV-6-3-coarse":
                        statistics = np.load(
                            f"{stats_path}/high-st-evadeV-6-3-coarse-comp-reg-stats-{x}.npy",
                            allow_pickle=True,
                        )
                    elif args.mc == "SnL-10x10":
                        statistics = np.load(
                            f"{stats_path}/high-st-SnL-coarse-comp-reg-stats-{x}.npy",
                            allow_pickle=True,
                        )
                    else:
                        statistics = np.load(
                            f"{stats_path}/high-st-{mc}-coarse-comp-reg-stats-{x}.npy",
                            allow_pickle=True,
                        )
                else:
                    if args.mc == "SnLw-10x10":
                        statistics = np.load(
                            f"{stats_path}/SnL-coarse-comp-reg-stats-{x}.npy",
                            allow_pickle=True,
                        )
                    elif args.mc == "evadeV-6-3-coarse":
                        statistics = np.load(
                            f"{stats_path}/evadeV-6-3-coarse-comp-reg-stats-{x}.npy",
                            allow_pickle=True,
                        )
                    else:
                        statistics = np.load(
                            f"{stats_path}/{mc}-coarse-comp-reg-stats-{x}.npy",
                            allow_pickle=True,
                        )
            else:
                if high_st:
                    statistics = np.load(
                        f"{stats_path}/high-st-{mc}-comp-reg-stats-{x}.npy",
                        allow_pickle=True,
                    )
                else:
                    statistics = np.load(
                        f"{stats_path}/{mc}-comp-reg-stats-{x}.npy",
                        allow_pickle=True,
                    )

        except FileNotFoundError:
            print(f"Statistics file for {x} not found, skipping.")
            continue

        obj = statistics.item()
        observations = obj["observations"]
        model_path = obj["args"]["model_path"]
        ohe = obj["one_hot_encoder"]

        reg_model = np.load(f"{model_path}.npy", allow_pickle=True).item()

        start_time = time.monotonic()
        reg_sub_trace = prep_traces_onehot_encoder(ohe, testing_samples, initial_amount)
        X = prep_traces_onehot_encoder(ohe, testing_samples, initial_amount)
        prep_time = time.monotonic() - start_time

        start_time = time.monotonic()
        reg_model.predict(X)
        prediction_time = time.monotonic() - start_time

        time_stats[x] = {
            "prep_time": prep_time,
            "prediction_time": prediction_time,
        }

    return time_stats


def aggreagted_stats_conformal(high_st, new_noisy, coarse, mc, model_path, stats_path):

    time_stats = {}

    for x in range(1, 11):
        try:
            if coarse:
                if high_st:
                    paths = glob.glob(
                        f"{model_path}/high-st-{mc}_coarse_comp_conformal_pred_state_estimator_{x}_*.pt"
                    )
                else:
                    paths = glob.glob(
                        f"{model_path}/{mc}_coarse_comp_conformal_pred_state_estimator_{x}_*.pt"
                    )
            else:
                if high_st:
                    paths = glob.glob(
                        f"{model_path}/high-st-{mc}_comp_conformal_pred_state_estimator_{x}_*.pt"
                    )
                else:
                    paths = glob.glob(
                        f"{model_path}/{mc}_comp_conformal_pred_state_estimator_{x}_*.pt"
                    )
        except FileNotFoundError:
            print(f"Statistics file for {x} not found, skipping.")
            continue

        try:
            if coarse:
                if high_st:
                    print(
                        f"{model_path}/high-st-{mc}_coarse_comp_conformal_pred_state_estimator_{x-1}_*.pt"
                    )
                    print(
                        "/workspaces/premise/out/models/2025-09-16_07-48-07/high-st-SnL-10x10_coarse_comp_conformal_pred_state_estimator_0_3775.pt"
                    )
                    state_estimator = torch.load(
                        glob.glob(
                            f"{model_path}/high-st-{mc}_coarse_comp_conformal_pred_state_estimator_{x-1}_*.pt"
                        )[0],
                        weights_only=False,
                    )
                    label_estimator = torch.load(
                        glob.glob(
                            f"{model_path}/high-st-{mc}_coarse_comp_conformal_pred_label_estimator_{x-1}_*.pt"
                        )[0],
                        weights_only=False,
                    )
                    cp_classification = torch.load(
                        glob.glob(
                            f"{model_path}/high-st-{mc}_coarse_comp_conformal_pred_cp_classification_{x-1}_*.pt"
                        )[0],
                        weights_only=False,
                    )

                    with open(
                        glob.glob(
                            f"{stats_path}/high-st-{mc}_coarse_comp_conformal_pred_rejection_classifier_{x-1}_*.pickle"
                        )[0],
                        "rb",
                    ) as f:
                        rej_classifier = pickle.load(f)

                    rejection_classifier = rej_classifier["rej_rule"]

                    with open(
                        glob.glob(
                            f"{stats_path}/high-st-{mc}_coarse_comp_conformal_pred_conformal_stats_{x-1}_*.pickle"
                        )[0],
                        "rb",
                    ) as f:
                        conformal_stats = pickle.load(f)
                else:
                    state_estimator = torch.load(
                        glob.glob(
                            f"{model_path}/{mc}_coarse_comp_conformal_pred_state_estimator_{x-1}_*.pt"
                        )[0],
                        weights_only=False,
                    )
                    label_estimator = torch.load(
                        glob.glob(
                            f"{model_path}/{mc}_coarse_comp_conformal_pred_label_estimator_{x-1}_*.pt"
                        )[0],
                        weights_only=False,
                    )
                    cp_classification = torch.load(
                        glob.glob(
                            f"{model_path}/{mc}_coarse_comp_conformal_pred_cp_classification_{x-1}_*.pt"
                        )[0],
                        weights_only=False,
                    )

                    with open(
                        glob.glob(
                            f"{stats_path}/{mc}_coarse_comp_conformal_pred_rejection_classifier_{x-1}_*.pickle"
                        )[0],
                        "rb",
                    ) as f:
                        rej_classifier = pickle.load(f)

                    rejection_classifier = rej_classifier["rej_rule"]

                    with open(
                        glob.glob(
                            f"{stats_path}/{mc}_coarse_comp_conformal_pred_conformal_stats_{x-1}_*.pickle"
                        )[0],
                        "rb",
                    ) as f:
                        conformal_stats = pickle.load(f)

            else:
                if high_st:
                    state_estimator = torch.load(
                        glob.glob(
                            f"{model_path}/high-st-{mc}_comp_conformal_pred_state_estimator_{x-1}_*.pt"
                        )[0],
                        weights_only=False,
                    )
                    label_estimator = torch.load(
                        glob.glob(
                            f"{model_path}/high-st-{mc}_comp_conformal_pred_label_estimator_{x-1}_*.pt"
                        )[0],
                        weights_only=False,
                    )
                    cp_classification = torch.load(
                        glob.glob(
                            f"{model_path}/high-st-{mc}_comp_conformal_pred_cp_classification_{x-1}_*.pt"
                        )[0],
                        weights_only=False,
                    )

                    with open(
                        glob.glob(
                            f"{stats_path}/high-st-{mc}_comp_conformal_pred_rejection_classifier_{x-1}_*.pickle"
                        )[0],
                        "rb",
                    ) as f:
                        rej_classifier = pickle.load(f)

                    rejection_classifier = rej_classifier["rej_rule"]

                    with open(
                        glob.glob(
                            f"{stats_path}/high-st-{mc}_comp_conformal_pred_conformal_stats_{x-1}_*.pickle"
                        )[0],
                        "rb",
                    ) as f:
                        conformal_stats = pickle.load(f)
                else:
                    state_estimator = torch.load(
                        glob.glob(
                            f"{model_path}/{mc}_comp_conformal_pred_state_estimator_{x-1}_*.pt"
                        )[0],
                        weights_only=False,
                    )
                    label_estimator = torch.load(
                        glob.glob(
                            f"{model_path}/{mc}_comp_conformal_pred_label_estimator_{x-1}_*.pt"
                        )[0],
                        weights_only=False,
                    )
                    cp_classification = torch.load(
                        glob.glob(
                            f"{model_path}/{mc}_comp_conformal_pred_cp_classification_{x-1}_*.pt"
                        )[0],
                        weights_only=False,
                    )

                    with open(
                        glob.glob(
                            f"{stats_path}/{mc}_comp_conformal_pred_rejection_classifier_{x-1}_*.pickle"
                        )[0],
                        "rb",
                    ) as f:
                        rej_classifier = pickle.load(f)

                    rejection_classifier = rej_classifier["rej_rule"]

                    with open(
                        glob.glob(
                            f"{stats_path}/{mc}_comp_conformal_pred_conformal_stats_{x-1}_*.pickle"
                        )[0],
                        "rb",
                    ) as f:
                        conformal_stats = pickle.load(f)

        except FileNotFoundError:
            print(f"Statistics file for {x-1} not found, skipping.")
            continue

        start_time = time.monotonic()
        new_noisy_scaled = -1 + 2 * (new_noisy - conformal_stats["dataset.MIN[1]"]) / (
            conformal_stats["dataset.MAX[1]"] - conformal_stats["dataset.MIN[1]"]
        )
        Y1 = np.transpose(new_noisy_scaled, (0, 2, 1))
        Y1t = Variable(FloatTensor(Y1))

        state_estimator.eval()
        state_estim = state_estimator(Y1t)
        label_estimator.eval()
        label_hypothesis = label_estimator(state_estim)

        label_prob = torch.nn.functional.softmax(label_hypothesis, dim=1)
        error_prob = label_prob[:, 1]
        error_prob = error_prob.tolist()

        pool_conf_cred = cp_classification.compute_confidence_credibility(
            np.transpose(new_noisy_scaled, (0, 2, 1))
        )
        keep_mask = utils.apply_svc_query_strategy(rejection_classifier, pool_conf_cred)

        for u in range(len(error_prob)):
            if keep_mask[u] == -1.0:
                error_prob[u] = 1.0
        conformal_time = time.monotonic() - start_time
        time_stats[x] = {
            "conformal_time": conformal_time,
        }

    return time_stats


def main_imc(args: argparse.Namespace):
    setup_logging("rq3:" + args.mc + str(args.coarse) + str(args.high_st))

    os.makedirs(args.out, exist_ok=True)

    suo, initial_amount, horizon = build_suo(args)

    length = initial_amount + horizon

    testing_samples = []
    for x in range(args.testing_samples):
        path = suo.generate_random_traces([], length)[0]
        testing_samples.append(tuple(path))

    models_dict = {"IP": InvertedPendulum(), "MC": mc_model(horizon)}
    model = models_dict["MC"]
    model = mc_model(horizon)

    noisy_measurements = model.get_noisy_measurments(testing_samples, horizon)

    alarms = aggregted_alarms(testing_samples)
    target_time_stats = stats_true(horizon, initial_amount, testing_samples, suo)

    coarse = args.coarse
    mc = args.mc
    model_path = args.model_path
    stats_path = args.stats_path
    high_st = args.high_st

    conformal_time_stats = aggreagted_stats_conformal(
        args.high_st, noisy_measurements, coarse, mc, model_path, stats_path
    )
    imc_time_stats = aggregated_stats_imc(
        args.high_st,
        coarse,
        "ref",
        mc,
        stats_path,
        initial_amount,
        horizon,
        args,
        testing_samples,
    )

    regression_time_stats = aggregated_stats_regression(
        args.high_st,
        coarse,
        mc,
        model_path,
        stats_path,
        testing_samples,
        horizon,
        initial_amount,
    )

    test_data = {}
    test_data["model"] = args.mc
    test_data["coarse"] = args.coarse
    test_data["high_st"] = args.high_st
    test_data["horizon"] = horizon
    test_data["initial_amount"] = initial_amount
    test_data["testing_samples"] = testing_samples
    test_data["alarms"] = alarms
    test_data["target_time_stats"] = target_time_stats
    test_data["imc_time_stats"] = imc_time_stats
    test_data["regression_time_stats"] = regression_time_stats
    test_data["conformal_time_stats"] = conformal_time_stats

    if args.coarse:
        if args.high_st:
            file_name = os.path.join(
                args.out, f"testdata_rq_4_{args.mc}_coarse_high_st.pkl"
            )
        else:
            file_name = os.path.join(args.out, f"testdata_rq_4_{args.mc}_coarse.pkl")
    else:
        if args.high_st:
            file_name = os.path.join(args.out, f"testdata_rq_4_{args.mc}_high_st.pkl")
        else:
            file_name = os.path.join(args.out, f"testdata_rq_4_{args.mc}.pkl")

    # Save dictionary
    with open(file_name, "wb") as f:
        pickle.dump(test_data, f)

    print(f"Time stats of {args.mc}:")
    print(f"Target monitor creation time: {target_time_stats['monitor_creation_time']}")
    print(
        f"Target sample time: mean {np.mean(target_time_stats['sample_times'])} std: {np.std(target_time_stats['sample_times'])}"
    )
    print(
        f"IMC monitor creation time: mean {np.mean([imc_time_stats[x]['monitor_creation_time'] for x in imc_time_stats])} std: {np.std([imc_time_stats[x]['monitor_creation_time'] for x in imc_time_stats])}"
    )
    print(
        f"IMC sample time: mean {np.mean([np.mean(imc_time_stats[x]['sample_times']) for x in imc_time_stats])} std: {np.std([np.mean(imc_time_stats[x]['sample_times']) for x in imc_time_stats])}"
    )
    print(
        f"Regression prep time: mean {np.mean([regression_time_stats[x]['prep_time'] for x in regression_time_stats])} std: {np.std([regression_time_stats[x]['prep_time'] for x in regression_time_stats])}"
    )
    print(
        f"Regression prediction time: mean {np.mean([regression_time_stats[x]['prediction_time'] for x in regression_time_stats])} std: {np.std([regression_time_stats[x]['prediction_time'] for x in regression_time_stats])}"
    )
    print(
        f"Conformal prediction time: mean {np.mean([conformal_time_stats[x]['conformal_time'] for x in conformal_time_stats])} std: {np.std([conformal_time_stats[x]['conformal_time'] for x in conformal_time_stats])}"
    )


def build_learning_parser(parser: argparse.ArgumentParser):
    group = parser.add_argument_group("Learning Parameters")

    group.add_argument(
        "--model_name",
        type=str,
        default="MC",
        help="Name of the conformal prediction model (first letters code).",
    )
    group.add_argument(
        "-s",
        "--testing_samples",
        type=int,
        default=500,
        help="Total number of samples used in learning",
    )
    group.add_argument(
        "--no-target",
        action="store_true",
        default=False,
        help="Do not use the target monitor",
    )


def testing_argsparser():
    parser = argparse.ArgumentParser(description="Learn an IMC")
    build_suo_args_parser(parser)
    build_learning_parser(parser)

    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="Increase verbosity level (can be used multiple times)",
    )

    parser.add_argument(
        "--model-path",
        type=str,
        help="Path to models",
    )

    parser.add_argument(
        "--stats-path",
        type=str,
        help="Path to stats",
    )

    parser.add_argument(
        "-c",
        "--coarse",
        action="store_true",
        default=False,
        help="If the leared model is coarse or not",
    )

    parser.add_argument(
        "-ht",
        "--high-st",
        action="store_true",
        default=False,
        help="If higher stopping threashold is used",
    )

    parser.add_argument(
        "-o",
        "--out",
        type=str,
        default="out/results",
        help="Output path for the results",
    )

    return parser


if __name__ == "__main__":
    parser = testing_argsparser()
    args = parser.parse_args()
    main_imc(args)


# python -m premise.interval.rq_3 --mc evadeV-5-3 --stats-path /workspaces/premise/out/stats/2025-08-01_08-37-30 --model-path /workspaces/premise/out/models/2025-08-01_08-37-30
