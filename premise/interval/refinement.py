import argparse
from typing import Any
import numpy as np
import stormpy.pomdp
import tqdm
import matplotlib.pyplot as plt
import pickle

from premise.interval.interval import create_monitor
from premise.models import build_model_and_risk, default_models
from premise.system import MCSystemUnderObservation, SystemUnderObservation
from premise.monitor import Monitor, PremiseOptions

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
        "--conformence-amount",
        type=int,
        default=500,
        help="Amount of extensions to generate for a sample",
    )
    parser.add_argument(
        "--horizon", required=True, type=int, help="The horizon to monitor on"
    )
    parser.add_argument(
        "--dump", type=str, help="Path to the file to dump the model to"
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

    refinement_results = {}
    x = []
    y1 = []
    y2 = []
    y3 = []

    testing_samples = []

    for i in range(50):  # Loop to run 25 times
        # Update paths for each iteration

        if i == 0:
            trans_path = "/workspaces/premise/premise/examples/SnL-10x10-interval.npy"
            init_path = (
                "/workspaces/premise/premise/examples/SnL-10x10-initial_interval.npy"
            )
        else:
            trans_path = (
                f"/workspaces/premise/premise/examples/SnL-10x10_{i+1}-interval.npy"
            )
            init_path = f"/workspaces/premise/premise/examples/SnL-10x10_{i+1}-initial_interval.npy"

        # Load learned model
        interval = np.load(trans_path, allow_pickle=True)[()]
        initial_interval = np.load(init_path, allow_pickle=True)[()]

        print(f"Running iteration {i+1} with:")
        print("Ready for conformance checking")

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

        # Run premise on the learned model
        samples = [ctr.generate_random_trace([], args.sample_length) for _ in range(50)]
        testing_samples.append(samples)
        risks: list[tuple[list, Any]] = []
        for trace in tqdm.tqdm(samples):
            mon.initialize(0)
            observations = [t[1] for t in trace]
            for obs in observations[:-1]:
                mon.step(observation_map[obs], compute_risk=False)
            last_risk = mon.step(observation_map[observations[-1]], compute_risk=True)
            risks.append((trace, last_risk))

        print("Monitoring learned model done")

        # Build the premise monitor on the real model
        expr_manager = stormpy.ExpressionManager()
        unfolder = stormpy.pomdp.create_observation_trace_unfolder(
            model, real_risk, expr_manager
        )
        ura = monitor.UnfoldingRiskAssessment(stormpy.Environment(), unfolder)
        mon = monitor.Monitor(ura, 1000000)

        # Run premise on the real model
        risks_real: list[tuple[list, Any]] = []
        for trace in tqdm.tqdm(samples):
            mon.initialize(trace[0][1])
            observations = [t[1] for t in trace]
            for obs in observations[1:-1]:
                mon.step(obs, compute_risk=False)
            last_risk = mon.step(observations[-1], compute_risk=True)
            risks_real.append((trace, last_risk))

        print("Monitoring done")

        # Check correctness of the monitor
        res = []
        for (trace, risk), (real_trace, real_risk) in zip(risks, risks_real):
            if trace != real_trace:
                raise ValueError("Traces do not match")

            alarm = 0
            for _ in range(args.conformence_amount):
                new_trace = ctr.generate_random_trace(
                    [s[1] for s in trace], length=len(trace) + args.horizon
                )
                alarm += any([s[2] for s in new_trace])
                if args.verbose > 0 and any([s[2] for s in new_trace]):
                    for s in new_trace:
                        if s[2]:
                            print("\033[92m", end="")
                        print(
                            model.state_valuations.get_string(s[0]).replace(" ", ""),
                            s[1],
                            end=" -> ",
                        )
                        if s[2]:
                            print("\033[0m", end="")
                    print()

            print(
                f"mon:{risk} real:{alarm / args.conformence_amount} real mon:{float(real_risk)} \n({[(model.state_valuations.get_string(s), o, b) for (s, o, b) in trace]})\n"
            )
            res.append((risk, alarm / args.conformence_amount, real_risk, trace))

            if args.verbose > 0:
                input()

        # Print statistics
        print("Results:")
        print(f"Total samples: {len(risks)}")
        print(f"Avg diff to sampling: {np.mean([abs(r[0] - r[1]) for r in res])}")
        print(f"Avg diff to premise: {np.mean([abs(r[0] - float(r[2])) for r in res])}")
        print(f"Best trace: {min(res, key=lambda x: abs(x[0] - x[1]))}")
        print(f"Worst trace: {max(res, key=lambda x: abs(x[0] - x[1]))}")

        conformance_failure = 0
        for r in res:
            if abs(r[0] - r[1]) > 0.01:
                conformance_failure += 1

        refinement_results[i] = (
            np.mean([abs(r[0] - r[1]) for r in res]),
            np.mean([abs(r[0] - float(r[2])) for r in res]),
            min(res, key=lambda x: abs(x[0] - x[1])),
            max(res, key=lambda x: abs(x[0] - x[1])),
            conformance_failure,
        )

        with open(
            "/workspaces/premise/premise/examples/refinement_results.pkl", "wb"
        ) as f:
            pickle.dump(refinement_results, f)

        print(f"Saved version: {x}")
