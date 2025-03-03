import argparse
import numpy as np
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

    # Load the real model into storm
    model, risk = models.build_model_and_risk(
        model_def,
        monitor.PremiseOptions(),
    )

    # Load learned model
    interval = np.load(args.trans_path, allow_pickle=True)[()]
    initial_interval = np.load(args.init_path, allow_pickle=True)[()]

    # Build the premise monitor
    mon, observation_map, unfolder, ipomdp = create_monitor(
        interval,
        initial_interval,
        "min",
        True,
        args.horizon,
        args.dump,
        args.verbose,
    )

    ctr = trace_generator.ConditionalTraceGenerator(
        model, target_label=model_def.target_label
    )

    print("Ready for conformance checking")

    # Run premise on the learned model
    samples = [ctr.generate_random_trace([], args.sample_length) for _ in range(50)]
    risks = []
    for trace in tqdm.tqdm(samples):
        mon.initialize(0)
        observations = [t[1] for t in trace]
        for obs in observations[:-1]:
            mon.step(observation_map[obs], compute_risk=False)
        last_risk = mon.step(observation_map[observations[-1]], compute_risk=True)
        risks.append((trace, last_risk))

    print("Monitoring done")

    # Check correctness of the monitor
    for trace, risk in risks:
        alarm = 0
        for _ in range(args.conformence_amount):
            new_trace = ctr.generate_random_trace(
                [s[1] for s in trace], length=len(trace) + args.horizon
            )
            alarm += any([s[2] for s in new_trace])
        print(
            f"mon:{risk} real:{risk / args.conformence_amount} ({[(model.state_valuations.get_string(s), model.observation_valuations.get_string(o), b) for (s, o, b) in trace]})"
        )
