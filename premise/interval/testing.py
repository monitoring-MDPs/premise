import argparse
import numpy as np
from tqdm import trange
import matplotlib.pyplot as plt
from sklearn import metrics

from premise.interval.conformence import test_monitor
from premise.interval.interval import create_monitor
from premise.models import default_models
from premise.system import MCSystemUnderObservation, SystemUnderObservation

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Conformance checking")

    model_group = parser.add_mutually_exclusive_group(required=True)
    model_group.add_argument(
        "-mc", "--mc", type=str, help="Use the premise model with the given name"
    )
    model_group.add_argument(
        "-sim", "--sim", type=str, help="Use the simulation model with the given name"
    )

    parser.add_argument(
        "-t",
        "--trans_path",
        required=True,
        type=str,
        help="Path to the transition dictionary",
    )
    parser.add_argument(
        "-i",
        "--init_path",
        required=True,
        type=str,
        help="Path to the initial state dictionary",
    )
    parser.add_argument(
        "-l",
        "--sample_length",
        required=True,
        type=int,
        help="Length of the samples to generate",
    )
    parser.add_argument(
        "-s", "--samples", type=int, default=1000, help="Amount of samples to test on"
    )
    parser.add_argument(
        "-ho", "--horizon", required=True, type=int, help="The horizon to monitor on"
    )
    parser.add_argument(
        "--dump", type=str, help="Path to the file to dump the model to"
    )

    parser.add_argument(
        "--dump-stats", type=str, help="Path to the file to dump stats to"
    )
    parser.add_argument("--verbose", "-v", action="count", default=0)

    args = parser.parse_args()

    if args.mc:
        model_def = default_models[args.mc]
        model_def.risk_property = (
            f'Pmax=? [F<={args.horizon} "{model_def.target_label}" ]'
        )
        suo: SystemUnderObservation = MCSystemUnderObservation(model_def, args.mc)

        if args.verbose > 1:
            for i, r in enumerate(suo.get_risk()):
                print(f"{suo._model.state_valuations.get_string(i)}: {float(r)}")
    elif args.sim:
        suo: SystemUnderObservation = None  # type: ignore
    else:
        raise ValueError("No model specified")

    # Load learned model
    interval = np.load(args.trans_path, allow_pickle=True)[()]
    initial_interval = np.load(args.init_path, allow_pickle=True)[()]

    print("Ready for testing")

    # Build the premise monitor on the learned model
    mon, observation_map, unfolder, ipomdp = create_monitor(
        interval,
        initial_interval,
        "min",
        True,
        args.horizon,
        args.dump,
        args.verbose,
    )

    alarms: list[bool] = []
    risks = []
    traces = []

    for x in trange(args.samples):

        trace = suo.generate_random_traces([], args.sample_length + args.horizon)[0]
        traces.append(trace)
        alarms.append(any([s[2] for s in trace]))

        # Run premise on the learned model
        sub_trace = trace[: args.sample_length]
        risk = test_monitor(
            mon,
            [sub_trace],
            obs_func=lambda x: observation_map[x],
            skip_initial=True,
            with_tqdm=False,
        )[sub_trace]

        risks.append(risk)

    if args.dump_stats:
        np.save(
            args.dump_stats,
            {
                "samples": traces,
                "alarms": alarms,
                "risks": risks,
            },  # type: ignore
        )

    print("Results:")
    print(f"Loaded {len(risks)} samples.")
    print(f"Alarms: {np.sum(alarms)} / {len(alarms)} ({100*np.mean(alarms):.2f}%)")
    print(
        f"Risks: min={np.min(risks):.4f}, max={np.max(risks):.4f}, mean={np.mean(risks):.4f}"
    )
    print(
        f"Risks: std={np.std(risks):.4f}, var={np.var(risks):.4f}, median={np.median(risks):.4f}"
    )

    # Print a trace with an alarm and without an alarm use the suo trace printer suo.trace_to_str(trace)
    print("Alarmed traces:")
    for i, trace in enumerate(traces):
        if alarms[i]:
            print(f"Trace {i}: {suo.trace_to_str(trace)}")
            print(f"Risk: {risks[i]}")
            print(
                f"True risk before horizon: {float(suo.get_risk()[trace[args.sample_length][0]])}"
            )
            print(f"True risk at end: {float(suo.get_risk()[trace[-1][0]])}")
            break
    print("Not alarmed traces:")
    for i, trace in enumerate(traces):
        if not alarms[i]:
            print(f"Trace {i}: {suo.trace_to_str(trace)}")
            print(f"Risk: {risks[i]}")
            print(
                f"True risk before horizon: {float(suo.get_risk()[trace[args.sample_length][0]])}"
            )
            print(f"True risk at end: {float(suo.get_risk()[trace[-1][0]])}")
            break
