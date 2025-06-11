import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from typing import Any
from sklearn.metrics import roc_curve, auc
from premise.monitor import Monitor
from sklearn.linear_model import LogisticRegression
import tqdm
import argparse
import matplotlib.pyplot as plt
from sklearn import metrics
from premise.interval.loading import build_suo, build_suo_args_parser
from premise.interval.loss import distance_measures
from premise.interval.interval import Samples, Trace, create_monitor
from premise.interval.learningIMC import learn_IMC
from premise.interval.conformence import test_monitor

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
        self.rnn = nn.GRU(input_size=input_dim, hidden_size=hidden_dim, batch_first=True)
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
        for s in x[:-args.horizon]:  # Exclude the horizon length
            training_trace.append(s[1]) 
        training_traces.append(training_trace)

    labels = []

    for x in train_samples:
        if any(s[2] == True for s in x[-args.horizon:]):
            labels.append(1)
        else: 
            labels.append(0)

    
    return training_traces, labels

def testing_data(test_samples, args):

    testing_traces = []
    for x in test_samples:
        testing_trace = []
        for s in x:  # Exclude the horizon length
            testing_trace.append(s[1]) 
        testing_traces.append(testing_trace)

    return testing_traces

def predict_on_test(model, test_traces, obs_to_idx, threshold=0.5):
    model.eval()
    vocab_size = len(obs_to_idx)
    predicted_probs = []

    with torch.no_grad():
        for trace in test_traces:
            mapped_trace = [obs_to_idx[o] for o in trace if o in obs_to_idx]
    
            trace_tensor = torch.zeros(len(mapped_trace), vocab_size)
            for i, idx in enumerate(mapped_trace):
                trace_tensor[i, idx] = 1.0

            trace_tensor = trace_tensor.unsqueeze(0) 
            prob = model(trace_tensor).item()

            predicted_probs.append(prob)

    return predicted_probs


def reg_main(args: argparse.Namespace):
    suo = build_suo(args)

    train_samples = suo.generate_random_traces(
        [], args.length + args.horizon, args.amount
    )

    test_samples = suo.generate_random_traces(
        [], args.length, args.test_samples
    )
    
    all_states, all_transitions, initial_states = suo.get_states_and_transitions() #all possible transitions
    all_observations = sorted({state[1] for state in all_states})  # Set for uniqueness
    obs_to_idx = {obs: idx for idx, obs in enumerate(all_observations)}
    vocab_size = len(obs_to_idx)

    traces, labels = training_data(train_samples, args)

    dataset = TraceDataset(traces, labels, vocab_size)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    model = TraceRNN(input_dim=vocab_size, hidden_dim=32)  
    train_model(model, dataloader, epochs=20)


    test_traces = testing_data(test_samples, args)
    predicted_probs = predict_on_test(model, test_traces, obs_to_idx)

    print(predicted_probs) 

    # Predict on the first trace
    model.eval()
    with torch.no_grad():
        sample, _ = dataset[0]
        sample = sample.unsqueeze(0)  # batch dim
        prob = model(sample).item()
        print(f"Predicted probability of error: {prob:.4f}")


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
        "-t", "--test_samples", type=int, default=5, help="Amount of test samples"
    )
    group.add_argument(
        "--model", type=bool, default=False, help="If a model exists"
    )
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
    reg_main(args)
    suo = build_suo(args)
