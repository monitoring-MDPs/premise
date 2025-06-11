import numpy as np
from sklearn.linear_model import LogisticRegression
import pandas as pd
import os
import pickle


alarms = []

stats = np.load('/workspaces/premise/results/stats/2025-05-11_18-07-23/airportA-7-10-10-comp-reg-stats.npy', allow_pickle=True).item()
observations = stats['observations']

traces = np.load("/workspaces/premise/premise/analysis/test_sets/airportA-7-10-10_l25_ho15.npy", allow_pickle=True)

traces = tuple(traces.tolist())

testing_traces = []
column_names = [f"Step{s}_Obs{o}" for s in range(25) for o in observations]

for s in traces:  

        t = tuple(x[1] for x in s)

        testing_traces.append(t)

        binary_test_data = []
        for trace in testing_traces:
            row = []
            for step in range(25):
                obs = trace[step]
                row.extend([1 if obs == o else 0 for o in observations])
            binary_test_data.append(row)

X_test = pd.DataFrame(binary_test_data, columns=column_names)


for trace in traces: 
    alarms.append(any([s[2] for s in trace]))

model = np.load('/workspaces/premise/results/models/2025-05-11_18-07-23/airportA-7-10-10-comp-reg.npy', allow_pickle=True).item()

prob = model.predict_proba(X_test)
risks = prob[:, 1]

stats = {
        "samples": traces,
        "alarms": alarms,
        "risks": risks,
        }

save_path = os.path.join("/workspaces/premise/premise/analysis/test_results_2025-05-11", "airportA-7-10-10_l25_ho15-reg.npy")
np.save(save_path, stats, allow_pickle=True)

