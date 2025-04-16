import argparse
import numpy as np
from tqdm import trange
import matplotlib.pyplot as plt
from sklearn import metrics

from premise.interval.interval import create_monitor
from premise.models import default_models
from system import MCSystemUnderObservation, SystemUnderObservation

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
        sub_trace = trace[: args.horizon]
        mon.initialize(0)
        observations = [s[1] for s in sub_trace]
        for obs in observations[:-1]:
            mon.step(observation_map[obs], compute_risk=False)
        last_risk = mon.step(observation_map[observations[-1]], compute_risk=True)
        risks.append(last_risk)

    # Print statistics
    print("Results:")

    np.save("/workspaces/premise/premise/examples/testing_samples.npy", trace)
    np.save("/workspaces/premise/premise/examples/risk_model_based.npy", risks)
    np.save("/workspaces/premise/premise/examples/alarms.npy", alarms)

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
