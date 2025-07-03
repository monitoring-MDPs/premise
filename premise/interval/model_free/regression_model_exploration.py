import numpy as np
import pandas as pd
from typing import Any
from sklearn.metrics import roc_curve, auc
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from premise.monitor import Monitor
from sklearn.linear_model import LogisticRegression

import tqdm
import argparse
import matplotlib.pyplot as plt
from sklearn import metrics

from premise.interval.loading import build_suo, build_suo_args_parser
from premise.interval.loss import distance_measures
from premise.interval.interval import Samples, Trace


class TraceDataset(Dataset):
    def __init__(self, traces, labels, vocab_size):
        self.traces = traces
        self.labels = labels
        self.vocab_size = vocab_size

    def __len__(self):
        return len(self.traces)

    def __getitem__(self, idx):
        trace = self.traces[idx]
        label = self.labels[idx]

        # Convert trace to one-hot vectors
        trace_tensor = torch.zeros(len(trace), self.vocab_size)
        for i, s in enumerate(trace):
            trace_tensor[i, s] = 1.0

        return trace_tensor, torch.tensor(label, dtype=torch.float32)


class TraceRNN(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(TraceRNN, self).__init__()
        self.rnn = nn.GRU(
            input_size=input_dim, hidden_size=hidden_dim, batch_first=True
        )
        self.fc = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        _, h_n = self.rnn(x)
        out = self.fc(h_n.squeeze(0))
        return self.sigmoid(out).squeeze(1)


def train_model(model, dataloader, epochs=20, lr=0.001):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCELoss()

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for traces, labels in dataloader:
            optimizer.zero_grad()
            outputs = model(traces)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss / len(dataloader):.4f}")


def training_data(train_samples, args):

    training_traces = []
    training_trace = []

    for x in train_samples:
        training_trace = []
        for s in x[: -args.horizon]:  # Exclude the horizon length
            training_trace.append(s[1])
        training_traces.append(training_trace)

    labels = []

    for x in train_samples:
        if any(s[2] == True for s in x[-args.horizon :]):
            labels.append(1)
        else:
            labels.append(0)

    return training_traces, labels


def testing_data(test_samples, args):

    testing_traces = []
    for x in test_samples:
        testing_trace = []
        for s in x[: -args.horizon]:  # Exclude the horizon length
            testing_trace.append(s[1])
        testing_traces.append(testing_trace)

    return testing_traces, test_samples


def predict_on_test(model, test_traces, test_samples, obs_to_idx, threshold=0.5):
    model.eval()
    vocab_size = len(obs_to_idx)
    predicted_probs = []

    predicted_probs = {}

    probabilities = []

    with torch.no_grad():
        for trace in test_traces:
            mapped_trace = [obs_to_idx[o] for o in trace if o in obs_to_idx]

            trace_tensor = torch.zeros(len(mapped_trace), vocab_size)
            for i, idx in enumerate(mapped_trace):
                trace_tensor[i, idx] = 1.0

            trace_tensor = trace_tensor.unsqueeze(0)
            prob = model(trace_tensor).item()
            probabilities.append(prob)

    for x in range(len(probabilities)):
        predicted_probs[test_samples[x]] = probabilities[x]

    return predicted_probs


# def learn_model_based(all_states, all_transitions, initial_states, train_samples, testing_samples):
#    min_width = 0.0
#    epsilon = 0.0001
#    initial_interval, interval = learn_IMC(all_states, all_transitions, initial_states, train_samples, min_width, epsilon)


# Build the premise monitor on the learned model
#    mon, observation_map, unfolder, ipomdp = create_monitor(
#        interval,
#        initial_interval,
#        "min",
#        True,
#        args.horizon,
#        dump_path=args.dump_model + "monitor" if args.dump_model else None,
#        verbose=args.verbose,
#    )

#    IMC_risks = test_monitor(
#        mon,
#        testing_samples,
#        obs_func=lambda x: observation_map[x],
#        skip_initial=True,
#    )
#
#    return IMC_risks


def learn_regression_model(train_samples, observations, testing_samples, args):
    X = []
    y = []

    for l in train_samples:  # path
        flattened_trace = []
        for x in l[: -args.horizon]:  # Exclude the horizon length
            flattened_trace.append(
                x[1]
            )  # Assuming x[1] contains the observable variables
        X.append(flattened_trace)

        # Determine the prediction based on the horizon
        if any(
            x[2] == True for x in l[-args.horizon :]
        ):  # Look at the last h steps for error state #CHECK THE LABEL NAMES
            y.append(1)
        else:
            y.append(0)

    num_steps = args.length

    column_names = [f"Step{s}_Obs{o}" for s in range(num_steps) for o in observations]
    print(f"Observation Count: {len(observations)}")
    print(f"Column Count: {len(column_names)}")

    binary_data = []
    for trace in X:
        row = []
        for step in range(num_steps):
            obs = trace[step]
            row.extend([1 if obs == o else 0 for o in observations])
        binary_data.append(row)

    X = pd.DataFrame(binary_data, columns=column_names)

    model = LogisticRegression()
    model.fit(X, y)

    testing_traces = []
    alarms = {}

    for s in testing_samples:
        t = tuple(x[1] for x in s[: -args.horizon])
        testing_traces.append(t)

    for s in testing_samples:
        alarms[tuple(s)] = any([x[2] for x in s[-args.horizon :]])

    binary_test_data = []
    for trace in testing_traces:
        row = []
        for step in range(num_steps):
            obs = trace[step]
            row.extend([1 if obs == o else 0 for o in observations])
        binary_test_data.append(row)

    X_test = pd.DataFrame(binary_test_data, columns=column_names)

    prob = model.predict_proba(X_test)

    risks = prob[:, 1]

    regression_risks = {}
    for sample, risk in zip(testing_samples, risks):
        regression_risks[tuple(sample)] = risk

    return testing_traces, alarms, regression_risks, model  # dictionary trace + risk


def test_regression_monitor(
    mon: Monitor,
    samples: Samples,
    obs_func=lambda x: x,
    skip_initial=False,
    with_tqdm=True,
):
    risks: dict[Trace, Any] = {}
    it = tqdm.tqdm(samples) if with_tqdm else samples
    for trace in it:
        observable = [t[1] for t in trace]

        if skip_initial:
            mon.initialize(0)
        else:
            mon.initialize(obs_func(observable[0]))

        for obser in observable[0 if skip_initial else 1 : -1]:
            mon.step(obs_func(obser), compute_risk=False)

        last_risk = mon.step(obs_func(observable[-1]), compute_risk=True)
        risks[tuple(trace)] = last_risk
    return risks


# def measure_distance(testing_samples, regression_risks, IMC_risks, distance, test_weights, suo, args):
def measure_distance(
    testing_samples, regression_risks, nn_risks, distance, test_weights, suo, args
):

    target_risks = test_regression_monitor(
        suo.create_target_monitor(),
        testing_samples,
    )

    target_dist_reg, target_all_dist_reg = distance.distance(
        test_weights,
        {s: float(r) for s, r in target_risks.items()},
        {s: float(r) for s, r in regression_risks.items()},
        all_distances=True,
    )

    target_dist_nn, target_all_dist_nn = distance.distance(
        test_weights,
        {s: float(r) for s, r in target_risks.items()},
        {s: float(r) for s, r in nn_risks.items()},
        all_distances=True,
    )

    # target_dist_imc, target_all_dist_imc = distance.distance(
    #    test_weights,
    #    {s: float(r) for s, r in target_risks.items()},
    #    {s: float(r) for s, r in IMC_risks.items()},
    #    all_distances=True,
    # )

    print(f"Regression's distance to target risk: {target_dist_reg}")
    print(f"Neural Network's distance to target risk: {nn_risks}")

    # print(f"IMC-based distance to target risk: {target_dist_imc}")

    # return target_risks, target_dist_reg, target_all_dist_reg, target_dist_imc, target_all_dist_imc
    return (
        target_risks,
        target_dist_reg,
        target_all_dist_reg,
        target_dist_nn,
        target_all_dist_nn,
    )


def plot_roc_curve(alarms, risks, fname=None):
    alarm_values = list(alarms.values())
    risk_values = list(risks.values())

    fpr, tpr, _ = roc_curve(alarm_values, risk_values)
    auc_value = auc(fpr, tpr)

    print(f"AUC: {auc_value:.4f}")

    # fpr, tpr, threshold = metrics.roc_curve(alarms, risks)
    # roc_auc = metrics.auc(fpr, tpr)

    # plt.figure(figsize=(8, 6))
    # plt.title("Receiver Operating Characteristic")
    # plt.plot(fpr, tpr, "b", label="AUC = %0.2f" % roc_auc)
    # plt.plot([0, 1], [0, 1], "r--")
    # plt.xlim((0, 1))
    # plt.ylim((0, 1))
    # plt.ylabel("True Positive Rate")
    # plt.xlabel("False Positive Rate")
    # plt.legend(loc="lower right")
    # plt.grid(True)
    # if fname:
    #    plt.savefig(fname)
    # plt.show()
    # print(f"AUC: {roc_auc:.4f}")
    return auc_value


# def monitoring_analysis(alarms, target_risks, regression_risks, IMC_risks):
def monitoring_analysis(alarms, target_risks, regression_risks, nn_risks):

    error = 0
    for x in alarms.keys():
        if alarms[x] == True:
            error += 1

    print(f"Error rate in samples: {error/len(alarms)}")

    print("Regression monitor")
    auc_value_reg = plot_roc_curve(alarms, regression_risks)

    print("Neural Network monitor")
    auc_value_nn = plot_roc_curve(alarms, nn_risks)

    # print("IMC monitor")
    # auc_value_imc = plot_roc_curve(alarms, IMC_risks)

    print("Target monitor")
    auc_value_target = plot_roc_curve(alarms, target_risks)

    # return auc_value_reg, auc_value_imc, auc_value_target
    return auc_value_reg, auc_value_nn, auc_value_target


def reg_main(args: argparse.Namespace):
    print(f"Learned on {args.amount} samples")

    suo, initial_amount, horizon = build_suo(args)

    if args.length is None:
        if initial_amount is not None:
            args.length = initial_amount
        else:
            raise ValueError(
                "Either length must be specified or initial_amount must be provided by the model."
            )

    if args.horizon is None:
        if horizon is not None:
            args.horizon = horizon
        else:
            raise ValueError(
                "Either horizon must be specified or it must be provided by the model."
            )

    train_samples = suo.generate_random_traces(
        [], args.length + args.horizon, args.amount
    )

    # all_states, all_transitions, initial_states = suo.get_states_and_transitions(
    #    all_transitions=not args.existing_transitions
    # )

    all_states, all_transitions, initial_states = (
        suo.get_states_and_transitions()
    )  # all possible transitions

    observations = []
    for x in all_states:
        if x[1] not in observations:
            observations.append(x[1])

    all_observations = sorted({state[1] for state in all_states})  # Set for uniqueness
    obs_to_idx = {obs: idx for idx, obs in enumerate(all_observations)}
    vocab_size = len(obs_to_idx)

    samples_with_prob = suo.generate_random_traces_with_prob(
        [], args.length + args.horizon, args.test_samples
    )

    total_prob = sum(p for _, p in samples_with_prob)
    test_weights = {s: float(p / total_prob) for s, p in samples_with_prob}
    testing_samples = [s[0] for s in samples_with_prob]

    traces, labels = training_data(train_samples, args)

    dataset = TraceDataset(traces, labels, vocab_size)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    model = TraceRNN(input_dim=vocab_size, hidden_dim=32)
    train_model(model, dataloader, epochs=20)

    test_traces, test_samples = testing_data(testing_samples, args)
    nn_risks = predict_on_test(model, test_traces, test_samples, obs_to_idx)

    distance = distance_measures[args.distance]()

    # IMC_risks = learn_model_based(
    #    all_states, all_transitions, initial_states, train_samples, testing_samples)

    testing_traces, alarms, regression_risks, model = learn_regression_model(
        train_samples, observations, testing_samples, args
    )

    # target_risks, target_dist_reg, target_all_dist_reg, target_dist_imc, target_all_dist_imc = measure_distance(
    #    testing_samples, regression_risks, IMC_risks, distance, test_weights, suo, args)

    (
        target_risks,
        target_dist_reg,
        target_all_dist_reg,
        target_dist_nn,
        target_all_dist_nn,
    ) = measure_distance(
        testing_samples, regression_risks, nn_risks, distance, test_weights, suo, args
    )

    # auc_value_reg, auc_value_imc, auc_value_target = monitoring_analysis(alarms, target_risks, regression_risks, IMC_risks)
    auc_value_reg, auc_value_nn, auc_value_target = monitoring_analysis(
        alarms, target_risks, regression_risks, nn_risks
    )

    if args.dump_stats:
        np.save(
            args.dump_stats,
            {
                "target_dist": target_dist,
                "target_all_dist": target_all_dist,
                "weights": {s: float(w) for s, w in test_weights.items()},
                "target_risks": {s: float(r) for s, r in target_risks.items()},
                "regression_risks": {s: float(r) for s, r in regression_risks.items()},
                "samples": testing_samples,
                "args": vars(args),
                "observations": observations,
            },  # type: ignore
        )

    if args.model_path:
        np.save(args.model_path, model)

    # return target_dist_reg, target_dist_imc, auc_value_reg, auc_value_imc, auc_value_target
    return (
        target_dist_reg,
        target_dist_nn,
        auc_value_reg,
        auc_value_nn,
        auc_value_target,
    )


def build_learning_args_parser(parser: argparse.ArgumentParser):
    group = parser.add_argument_group("Learning Parameters")
    group.add_argument(
        "-m", "--model-path", type=str, default=None, help="Path to store the model"
    )
    group.add_argument(
        "-a", "--amount", type=int, default=1000, help="Amount of samples to generate"
    )
    group.add_argument(
        "-l", "--length", type=int, default=15, help="Length of the samples to generate"
    )
    group.add_argument(
        "-t", "--test_samples", type=int, default=500, help="Amount of test samples"
    )
    group.add_argument("--model", type=bool, default=False, help="If a model exists")
    group.add_argument("--horizon", type=int, default=5, help="Length horizon")


def reg_argsparser():
    parser = argparse.ArgumentParser(description="Learn an IMC")
    build_suo_args_parser(parser)
    build_learning_args_parser(parser)
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="Increase verbosity level (can be used multiple times)",
    )
    parser.add_argument(
        "-d",
        "--distance",
        choices=distance_measures.keys(),
        default="mae",
        help="Distance measure to use",
    )

    parser.add_argument("--dump-model", type=str, help="Path to dump the model to")
    parser.add_argument(
        "--dump-stats", type=str, help="Path to the file to dump stats to"
    )

    return parser


if __name__ == "__main__":
    parser = reg_argsparser()
    args = parser.parse_args()

    model_free_distance = []
    # model_based_distance = []
    nn_distance = []
    model_free_AUC = []
    # model_based_AUC = []
    nn_AUC = []
    target_AUC = []

    for x in range(250, 20250, 250):
        args.amount = x
        # target_dist_reg, target_dist_imc, auc_value_reg, auc_value_imc, auc_value_target  = reg_main(args)
        (
            target_dist_reg,
            target_dist_nn,
            auc_value_reg,
            auc_value_nn,
            auc_value_target,
        ) = reg_main(args)

        model_free_distance.append(target_dist_reg)
        # model_based_distance.append(target_dist_imc)
        nn_distance.append(target_dist_nn)

        model_free_AUC.append(auc_value_reg)
        # model_based_AUC.append(auc_value_imc)
        nn_AUC.append(auc_value_nn)
        target_AUC.append(auc_value_target)

        args = parser.parse_args()
        suo = build_suo(args)

    print(model_free_distance)
    # print(model_based_distance)
    print(nn_distance)
    print(target_AUC)
    print(model_free_AUC)
    print(nn_AUC)
    # print(model_based_AUC)
