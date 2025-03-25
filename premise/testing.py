import argparse
from typing import Any
import numpy as np
import tqdm
import matplotlib.pyplot as plt
from sklearn import metrics

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
        "-s", "--samples", type=int, default=1000, help="Amount of samples to test on"
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

    print("Ready for tseting")

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

    for x in tqdm.trange(args.samples):

        trace = ctr.generate_random_trace([], args.sample_length)
        traces.append(trace)

        # Run premise on the learned model
        mon.initialize(0)
        observations = [s[1] for s in trace]
        for obs in observations[:-1]:
            mon.step(observation_map[obs], compute_risk=False)
        last_risk = mon.step(observation_map[observations[-1]], compute_risk=True)
        risks.append(last_risk)

        # Check correctness of the monitor
        new_trace = ctr.generate_random_trace(
            [s[1] for s in trace], length=len(trace) + args.horizon
        )
        alarms.append(any([s[2] for s in new_trace]))

        if args.verbose > 0:
            input()

    # Print statistics
    print("Results:")

    not_reached = 0
    reached = 0
    false_negative = 0
    false_positve = 0

    np.save("/workspaces/premise/premise/examples/testing_samples.npy", trace)
    np.save("/workspaces/premise/premise/examples/risk_model_based.npy", risks)
    np.save("/workspaces/premise/premise/examples/alarms.npy", alarms)

    for x in range(args.samples):
        if not alarms[x]:
            not_reached += 1
            if risks[x] > (0.03):
                false_positve += 1
        if alarms[x]:
            reached += 1
            if risks[x] < (0.03):
                false_negative += 1

    print(f"Not reached : {not_reached/args.samples}")
    print(f"Reached : {reached/args.samples}")
    print(f"False Positives: {false_positve/not_reached}")
    print(f"False Negatives: {false_negative/reached}")
    print(f"Error rate: {(false_negative + false_positve)/ args.samples}")

    fpr, tpr, threshold = metrics.roc_curve(alarms, risks)
    roc_auc = metrics.auc(fpr, tpr)

    fig, axis = plt.subplots(1, 1, figsize=(8, 6))

    axis.set_title("Receiver Operating Characteristic")
    axis.plot(fpr, tpr, "b", label="AUC = %0.2f" % roc_auc)
    axis.legend(loc="lower right")
    axis.plot([0, 1], [0, 1], "r--")
    axis.set_xlim((0, 1))
    axis.set_ylim((0, 1))
    axis.set_ylabel("True Positive Rate")
    axis.set_xlabel("False Positive Rate")

    fig.set_size_inches(18.5, 10.5)
    plt.savefig("res.png")
