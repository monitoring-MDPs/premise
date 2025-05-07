import numpy as np
import sklearn
import pickle

import pandas as pd
from sklearn.linear_model import LogisticRegression

#leaning_samples = np.load("/workspaces/premise/premise/examples/learning_samples.npy")

with open('/workspaces/premise/premise/carla/carla_samples/all_data.pkl', "rb") as f:
    learning_samples = pickle.load(f)


X = []
y = []

for l in learning_samples:  # 1000 traces 
    #for s in learning_samples[l]:  # s - state
        # Flatten the trace and append to training_data
        flattened_trace = []
        for x in l[:-225]:  # Exclude the last 200 states (the monitor is given the first 50)
            flattened_trace.append(x[1])  # Assuming x[1] contains the feature value (observation)
        X.append(flattened_trace)

        # Determine the prediction based on the last 5 elements in the trace
        if any(x[2] == True for x in l[-225:-150]): # horizon of 75
            y.append(1)
        else:
            y.append(0)
    

num_steps = 25


colors = ['LLL', 'LLM', 'LLH', 'LML', 'LMM', 'LMH', 'LHL', 'LHM', 'LHH',
 'MLL', 'MLM', 'MLH', 'MML', 'MMM', 'MMH', 'MHL', 'MHM', 'MHH',
 'HLL', 'HLM', 'HLH', 'HML', 'HMM', 'HMH', 'HHL', 'HHM', 'HHH']

percived_distance = ['pd10', 'pd20','pd30','pd40','pd50','pd60','pd70']

observations = []
for a in colors: 
    for b in percived_distance: 
        observations.append((a,b))


column_names = [f"Step{s}_Obs{o}" for s in range(num_steps) for o in observations]

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

#testing_samples = np.load("/workspaces/premise/premise/examples/testing_samples.npy")
X_test =  [[('HHH', 'pd60')] * 25]

binary_test_data = []
for trace in X_test:
    row = []
    for step in range(num_steps):
        obs = trace[step]
        row.extend([1 if obs == o else 0 for o in observations])
    binary_test_data.append(row)


X_test = pd.DataFrame(binary_test_data, columns=column_names)


# risk = []
# prob = decision_tree.predict_proba([row.values])[0][1]  # Probability of class 1 (error)
prob = model.predict_proba(X_test)
print(prob)
# decision_tree.predict_proba(pd.DataFrame([row], columns=column_names))[0][1]
# risk.append(prob)

prob_class_1 = prob[:, 1]

print(prob_class_1)

#alarms = np.load("/workspaces/premise/premise/examples/alarms.npy")
#risk_model_based = np.load("/workspaces/premise/premise/examples/risk_model_based.npy")


