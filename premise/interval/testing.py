import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
import os
import pickle
from stormpy import AddUncertaintyExact, export_to_drn, Rational

from premise.system import SystemUnderObservation, MCSystemUnderObservation
from premise.monitor import PremiseOptions
from premise.models import default_models
from premise.interval.interval import (
    stormpy_imdp_to_ipomdp,
    stormpy_exact_pomdp_to_mdp,
)
from premise.interval.regression_model import prep_trace_for_regression
from premise.interval.loading import (
    build_imc_loading_args_parser,
    build_suo,
    build_suo_args_parser,
    load_imc,
)
from premise.interval.conformence import random_sample_monitor_test, test_monitor
from premise.interval.interval import (
    Samples,
    Trace,
    create_monitor,
    build_monitor_from_model,
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Conformance checking")

    build_suo_args_parser(parser)
    build_imc_loading_args_parser(parser, required=False)

    parser.add_argument(
        "-reg",
        "--regression-path",
        type=str,
        default=None,
        help="Path to the regression model",
    )
    parser.add_argument(
        "-ro",
        "--regression-observations",
        type=str,
        default=None,
        help="Observations to use for the regression model",
    )

    parser.add_argument(
        "-l",
        "--sample_length",
        required=True,
        type=int,
        help="Length of the samples to test on (without the horizon)",
    )
    parser.add_argument(
        "-s", "--samples", type=int, default=100, help="Amount of samples to test on"
    )
    parser.add_argument(
        "-ts", "--test_data_set", type=str, help="Path to a set of test cases"
    )
    parser.add_argument(
        "-ho", "--horizon", required=True, type=int, help="The horizon to monitor on"
    )
    parser.add_argument(
        "--no-target",
        action="store_true",
        default=False,
        help="Do not use the target monitor",
    )
    parser.add_argument(
        "-au",
        "--additive-uncertainty",
        nargs="+",
        help="Test on target monitor with specified additive uncertainty, can be given mulitple arguments for multiple monitors. Target monitor has to be turned on",
    )
    parser.add_argument(
        "-e",
        "--exact",
        action="store_true",
        help="Use exact conformance checking",
    )
    parser.add_argument(
        "-p",
        "--precision",
        type=float,
        default=1e-6,
        help="Precision to use for the exact conformance checking",
    )
    parser.add_argument(
        "--sampling-amount",
        type=int,
        default=None,
        help="Turn on sampling with this many extensions",
    )
    parser.add_argument(
        "--dump", type=str, help="Path to the file to dump the model to"
    )

    parser.add_argument(
        "--dump-stats",
        type=str,
        default=None,
        help="Path to the file to dump stats to (leave empty to dump to defined path)",
    )
    parser.add_argument("--verbose", "-v", action="count", default=0)

    args = parser.parse_args()

    suo = build_suo(args)

    states, trans, initial = suo.get_states_and_transitions(False)
    print(
        f"Testing on system with {len(states)} ({len(initial)} initial) and {len(trans)} transitions."
    )

    has_imc = args.trans_path is not None and args.init_path is not None
    if has_imc:
        interval, initial_interval = load_imc(args)
        # Build the premise monitor on the learned model
        mon, observation_map, unfolder, ipomdp = create_monitor(
            interval,
            initial_interval,
            "min",
            True,
            args.horizon,
            args.dump,
            args.verbose,
            use_exact=args.exact,
            precision=args.precision,
        )

    # Load the regression model
    if args.regression_path and args.regression_observations:
        reg_model = np.load(args.regression_path, allow_pickle=True).item()
        observations = np.load(args.regression_observations, allow_pickle=True).item()[
            "observations"
        ]
        column_names = [
            f"Step{s}_Obs{o}" for s in range(args.sample_length) for o in observations
        ]
    else:
        reg_model = None

    uncertain_monitors = None
    if not args.no_target:
        target_monitor = suo.create_target_monitor()

        model_def = default_models[args.mc]
        model_def.risk_property = (
            f'Pmax=? [F<={vars(args).get("horizon", 1)} "{model_def.target_label}" ]'
        )
        non_exact_suo: SystemUnderObservation = MCSystemUnderObservation(
            model_def, args.mc, PremiseOptions(exact_arithmetic=False)
        )
        float_target_monitor = non_exact_suo.create_target_monitor()

        if (
            args.additive_uncertaint
            and "_model" in suo.__dict__
            and "_model_def" in suo.__dict__
        ):
            uncertain_monitors = {}
            pomdp = suo._model
            mdp = stormpy_exact_pomdp_to_mdp(suo._model)
            au_transformer = AddUncertaintyExact(mdp)
            for au in args.additive_uncertainty:
                imdp = au_transformer.transform(Rational(float(au)), Rational(0.0001))
                ipomdp = stormpy_imdp_to_ipomdp(
                    imdp, pomdp.observations, pomdp.observation_valuations
                )
                au_mon, _ = build_monitor_from_model(
                    ipomdp,
                    "min",
                    args.horizon,
                    target=suo._model_def.target_label,
                )
                uncertain_monitors[au] = au_mon

    print("Ready for testing")

    if args.test_data_set:
        traces: Samples = tuple(tuple(s) for s in np.load(args.test_data_set, allow_pickle=True).tolist())  # type: ignore
    else:
        traces = suo.generate_random_traces(
            [], args.sample_length + args.horizon, args.samples
        )

    alarms: list[bool] = []
    imc_risks = []
    target_risks = []
    float_target_risks = []
    regression_risks = []
    sampling_risks = []
    uncertain_risks = {}

    if uncertain_monitors is not None:
        for au, _ in uncertain_monitors.items():
            uncertain_risks[au] = []

    for trace in tqdm(traces):
        alarms.append(any([s[2] for s in trace]))
        sub_trace: Trace = trace[: args.sample_length]

        # Run the target monitor
        if not args.no_target:
            target_risk = test_monitor(
                target_monitor,
                [sub_trace],
                with_tqdm=False,
            )
            target_risk = target_risk[sub_trace]
            target_risks.append(float(target_risk))

            float_target_risk = test_monitor(
                float_target_monitor,
                [sub_trace],
                with_tqdm=False,
            )
            float_target_risk = float_target_risk[sub_trace]
            float_target_risks.append(float(float_target_risk))

        if uncertain_monitors is not None:
            for au, au_mon in uncertain_monitors.items():
                uncertain_risk = test_monitor(
                    au_mon,
                    [sub_trace],
                    with_tqdm=False,
                )
                uncertain_risk = uncertain_risk[sub_trace]
                uncertain_risks[au].append(float(uncertain_risk))

        # Run premise on the learned model
        if has_imc:
            risk = test_monitor(
                mon,
                [sub_trace],
                obs_func=lambda x: observation_map[x],
                skip_initial=True,
                with_tqdm=False,
            )[sub_trace]

            imc_risks.append(risk)

        # Run regression model
        if reg_model:
            reg_sub_trace = prep_trace_for_regression(sub_trace, observations)
            X = pd.DataFrame([reg_sub_trace], columns=column_names)
            prob = reg_model.predict_proba(X)
            regression_risks.append(prob[0, 1])

        if args.sampling_amount is not None:
            sampling_risk = random_sample_monitor_test(
                suo,
                [sub_trace],
                args.horizon,
                args.sampling_amount,
                with_tqdm=False,
            )[sub_trace]
            sampling_risks.append(sampling_risk)

    if args.dump_stats is not None:
        stats = {"args": vars(args), "samples": traces, "alarms": alarms, "risks": {}}
        if has_imc:
            stats["risks"]["imc_risks"] = imc_risks

        if not args.no_target:
            stats["risks"]["target_risks"] = target_risks
            stats["risks"]["float_target_risks"] = float_target_risks

        if reg_model:
            stats["risks"]["regression_risks"] = regression_risks

        if args.sampling_amount is not None:
            stats["risks"]["sampling_risks"] = sampling_risks

        if uncertain_monitors is not None:
            for au, risks in uncertain_risks.items():
                stats["risks"][f"uncertain_risks_{au}"] = risks

        if args.dump_stats == "":
            filename = os.path.join(
                args.dump_stats, f"{args.mc}_l{args.sample_length}_ho{args.horizon}.npy"
            )
            os.makedirs(os.path.dirname(filename), exist_ok=True)
        else:
            filename = args.dump_stats

        with open(filename, "wb") as f:
            pickle.dump(stats, f)

    print("Results:")
    print(f"Loaded {len(traces)} samples.")
    print(f"Alarms: {np.sum(alarms)} / {len(alarms)} ({100*np.mean(alarms):.2f}%)")

    if has_imc:
        print(
            f"IMC Risks: min={np.min(imc_risks):.4f}, max={np.max(imc_risks):.4f}, mean={np.mean(imc_risks):.4f} "
            f"std={np.std(imc_risks):.4f}, var={np.var(imc_risks):.4f}, median={np.median(imc_risks):.4f}"
        )

    if not args.no_target:
        print(
            f"Target risks: min={np.min(target_risks):.4f}, max={np.max(target_risks):.4f}, mean={np.mean(target_risks):.4f} "
            f"std={np.std(target_risks):.4f}, var={np.var(target_risks):.4f}, median={np.median(target_risks):.4f}"
        )

    if reg_model:
        print(
            f"Regression risks: min={np.min(regression_risks):.4f}, max={np.max(regression_risks):.4f}, mean={np.mean(regression_risks):.4f} "
            f"std={np.std(regression_risks):.4f}, var={np.var(regression_risks):.4f}, median={np.median(regression_risks):.4f}"
        )

    if args.sampling_amount is not None:
        print(
            f"Sampling risks: min={np.min(sampling_risks):.4f}, max={np.max(sampling_risks):.4f}, mean={np.mean(sampling_risks):.4f} "
            f"std={np.std(sampling_risks):.4f}, var={np.var(sampling_risks):.4f}, median={np.median(sampling_risks):.4f}"
        )

    # Print a trace with an alarm and without an alarm use the suo trace printer suo.trace_to_str(trace)
    print("Alarmed traces:")
    for i, trace in enumerate(traces):
        if alarms[i]:
            print(f"Trace {i}: {suo.trace_to_str(trace[: args.sample_length])}")
            print(f"Horizon: {suo.trace_to_str(trace[args.sample_length:])}")
            if has_imc:
                print(f"IMC Risk: {imc_risks[i]}")
            if not args.no_target:
                print(f"Target risk: {target_risks[i]}")
            if reg_model:
                print(f"Regression risk: {regression_risks[i]}")
            if args.sampling_amount is not None:
                print(f"Sampling risk: {sampling_risks[i]}")

            break

    print("Not alarmed traces:")
    for i, trace in enumerate(traces):
        if not alarms[i]:
            print(f"Trace {i}: {suo.trace_to_str(trace[: args.sample_length])}")
            print(f"Horizon: {suo.trace_to_str(trace[args.sample_length:])}")
            if has_imc:
                print(f"IMC Risk: {imc_risks[i]}")
            if not args.no_target:
                print(f"Target risk: {target_risks[i]}")
            if reg_model:
                print(f"Regression risk: {regression_risks[i]}")
            if args.sampling_amount is not None:
                print(f"Sampling risk: {sampling_risks[i]}")

            break
