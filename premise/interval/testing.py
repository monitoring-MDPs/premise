import argparse
import numpy as np
from tqdm import trange
import os
import pickle

from premise.interval.loading import (
    build_imc_loading_args_parser,
    build_suo,
    build_suo_args_parser,
    load_imc,
)
from premise.interval.conformence import test_monitor
from premise.interval.interval import create_monitor

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Conformance checking")

    build_suo_args_parser(parser)
    build_imc_loading_args_parser(parser)

    parser.add_argument(
        "-l",
        "--sample_length",
        required=True,
        type=int,
        help="Length of the samples to generate",
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
        "--no-target", action="store_true", help="Do not use the target monitor"
    )
    parser.add_argument(
        "--dump", type=str, help="Path to the file to dump the model to"
    )

    parser.add_argument(
        "--dump-stats", type=str, help="Path to the file to dump stats to"
    )
    parser.add_argument("--verbose", "-v", action="count", default=0)
    parser.add_argument("--premise", "-p", type=bool, default=True, help='Is the monior an IMC meant as Premise input')

    args = parser.parse_args()

    suo = build_suo(args)

    if args.premise == True: 

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
        )


    if not args.no_target:
        target_monitor = suo.create_target_monitor()

    alarms: list[bool] = []
    risks = []
    target_risks = []
    traces = []

    print("Ready for testing")

    if args.test_data_set: 
        traces = np.load(args.test_data_set, allow_pickle=True)
        traces = tuple(traces.tolist())
        print(type(traces))

    else:
        for x in trange(args.samples):

            trace = suo.generate_random_traces([], args.sample_length + args.horizon)[0]
            traces.append(trace)
        
    for trace in traces: 

        alarms.append(any([s[2] for s in trace]))

        if args.premise == True: 

            # Run premise on the learned model
            sub_trace = tuple(tuple(step) for step in trace[:args.sample_length])
            print(type(sub_trace[0]))
            risk = test_monitor(
                mon,
                [sub_trace],
                obs_func=lambda x: observation_map[x],
                skip_initial=True,
                with_tqdm=False,
            )[sub_trace]

            risks.append(risk)

            if not args.no_target:
                target_risk = test_monitor(
                    target_monitor,
                    [sub_trace],
                    with_tqdm=False,
                )[sub_trace]
                target_risks.append(target_risk)

        else: 
            model = np.load('/workspaces/premise/results/models/2025-05-10_12-12-21/airportA-7-10-10-comp-reg.npy', allow_pickle=True)
            prob = model.predict_proba(traces)

            risks = prob[:, 1]

  
    if args.dump_stats:
        stats = {
            "samples": traces,
            "alarms": alarms,
            "risks": risks,
        }
        if not args.no_target:
            stats["target_risks"] = target_risks
        
        filename = os.path.join(args.dump_stats, f"{args.mc}_l{args.sample_length}_ho{args.horizon}.npy")
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, 'wb') as f:
            pickle.dump(stats, f)

       

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
            if not args.no_target:
                print(f"Target risk: {target_risks[i]}")
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
            if not args.no_target:
                print(f"Target risk: {target_risks[i]}")
                print(
                    f"True risk before horizon: {float(suo.get_risk()[trace[args.sample_length][0]])}"
                )
                print(f"True risk at end: {float(suo.get_risk()[trace[-1][0]])}")
            break





#python -m premise.interval.testing -mc airportA-7-10-10 -l 25 -ho 15 --no-target --trans_path /workspaces/premise/results/models/2025-05-11_18-07-23/airportA-7-10-10-interval.npy --init_path /workspaces/premise/results/models/2025-05-11_18-07-23/airportA-7-10-10-initial_interval.npy --dump-stats /workspaces/premise/premise/analysis/test_results_2025-05-11 -ts /workspaces/premise/premise/analysis/test_sets/airportA-7-10-10_l25_ho15.npy
#python -m premise.interval.testing -mc airportA-7-10-10 -l 25 -ho 15 --no-target --trans_path  /workspaces/premise/results/models/2025-05-11_18-07-23/airportA-7-10-10-comp-no-ref-interval.npy --init_path /workspaces/premise/results/models/2025-05-11_18-07-23/airportA-7-10-10-comp-no-ref-initial_interval.npy --dump-stats /workspaces/premise/premise/analysis/test_results/test_results_2025-05-11 -ts /workspaces/premise/premise/analysis/test_sets/airportA-7-10-10_l25_ho15.npy

#python -m premise.interval.testing -mc SnL-10x10 -l 15 -ho 5 --no-target --trans_path /workspaces/premise/results/models/2025-05-11_18-07-23/SnL-10x10-comp-interval.npy  --init_path /workspaces/premise/results/models/2025-05-11_18-07-23/SnL-10x10-comp-initial_interval.npy --dump-stats /workspaces/premise/premise/analysis/test_results/test_results_2025-05-11 -ts /workspaces/premise/premise/analysis/test_sets/SnL-10x10_l15_ho5.npy
#python -m premise.interval.testing -mc SnL-10x10 -l 15 -ho 5 --no-target --trans_path /workspaces/premise/results/models/2025-05-11_18-07-23/SnL-10x10-comp-no-ref-interval.npy /workspaces/premise/results/models/2025-05-11_18-07-23/SnL-10x10-comp-no-ref-initial_interval.npy --init_path --dump-stats /workspaces/premise/premise/analysis/test_results/test_results_2025-05-11 -ts /workspaces/premise/premise/analysis/test_sets/SnL-10x10_l15_ho5.npy

#python -m premise.interval.testing -mc evadeV-6-3 -l 20 -ho 12 --no-target --trans_path /workspaces/premise/results/models/2025-05-10_12-12-21/evadeV-interval.npy --init_path /workspaces/premise/results/models/2025-05-10_12-12-21/evadeV-initial_interval.npy --dump-stats /workspaces/premise/premise/analysis/test_results -ts /workspaces/premise/premise/analysis/test_sets/evadeV-6-3_l20_ho12.npy
#python -m premise.interval.testing -mc evadeV-6-3 -l 20 -ho 12 --no-target --trans_path /workspaces/premise/results/models/2025-05-10_12-12-21/evadeV-comp-no-ref-interval.npy --init_path  /workspaces/premise/results/models/2025-05-10_12-12-21/evadeV-comp-no-ref-initial_interval.npy --dump-stats /workspaces/premise/premise/analysis/test_results -ts /workspaces/premise/premise/analysis/test_sets/evadeV-6-3_l20_ho12.npy


#python -m premise.interval.testing -mc -l -ho  --no-target --trans_path --init_path  --dump-stats /workspaces/premise/premise/analysis/test_results -ts
#python -m premise.interval.testing -mc  -l -ho  --no-target --trans_path --init_path  --dump-stats /workspaces/premise/premise/analysis/test_results -ts

#python -m premise.interval.testing -mc  -l -ho  --no-target --trans_path --init_path  --dump-stats /workspaces/premise/premise/analysis/test_results -ts
#python -m premise.interval.testing -mc  -l -ho  --no-target --trans_path --init_path  --dump-stats /workspaces/premise/premise/analysis/test_results -ts


#python -m premise.interval.testing -mc airportA-7-10-10 -l 25 -ho 15 --no-target --dump-stats /workspaces/premise/premise/analysis/test_results -ts /workspaces/premise/premise/analysis/test_sets/airportA-7-10-10_l25_ho15.npy


