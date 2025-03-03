import argparse
from numbers import Number
from typing import Any
import numpy as np
import stormpy.pomdp
import tqdm

from interval import create_monitor
import models
import monitor
import trace_generator


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Conformance checking")
    parser.add_argument(
        "-m",
        "--model_name",
        type=str,
        default="SnL-10x10",
        help="Name of the model to use",
    )
    parser.add_argument(
        "-t", "--trans_path", type=str, help="Path to the transition dictionary"
    )
    parser.add_argument(
        "-i", "--init_path", type=str, help="Path to the initial state dictionary"
    )
    parser.add_argument(
        "-l", "--sample_length", type=int, help="Length of the samples to generate"
    )
    parser.add_argument(
        "--conformence-amount",
        type=int,
        default=500,
        help="Amount of extensions to generate for a sample",
    )
    parser.add_argument("--horizon", type=int, help="The horizon to monitor on")
    parser.add_argument(
        "--dump", type=str, help="Path to the file to dump the model to"
    )
    parser.add_argument("--verbose", "-v", action="count", default=0)

    args = parser.parse_args()

    # Get the real model definition
    model_def = models.default_models[args.model_name]

    model_def.risk_property = f'Pmax=? [F<={args.horizon} "{model_def.target_label}" ]'

    # Load the real model into storm
    model, real_risk = models.build_model_and_risk(
        model_def,
        monitor.PremiseOptions(),
    )

    if args.verbose > 0:
        for i, r in enumerate(real_risk):
            print(f"{model.state_valuations.get_string(i)}: {float(r)}")

    # Load learned model
    interval = np.load(args.trans_path, allow_pickle=True)[()]
    initial_interval = np.load(args.init_path, allow_pickle=True)[()]

    ctr = trace_generator.ConditionalTraceGenerator(
        model, target_label=model_def.target_label
    )

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
