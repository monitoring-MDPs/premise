import numpy as np
import sklearn
import pickle

import pandas as pd
from sklearn.linear_model import LogisticRegression


with open('/workspaces/premise/premise/carla/carla_samples/all_data_more.pkl', "rb") as f:
    learning_samples = pickle.load(f)


with open('/workspaces/premise/premise/carla/carla_samples/all_data.pkl', "rb") as f:
    base_data = pickle.load(f)

with open('/workspaces/premise/premise/carla/carla_samples/HHH_0_50.pkl', "rb") as f:
    HHH_data_1 = pickle.load(f)

with open('/workspaces/premise/premise/carla/carla_samples/HHH_0_50.pkl', "rb") as f:
    HHH_data_2 = pickle.load(f)

with open('/workspaces/premise/premise/carla/carla_samples/HHH_0_50.pkl', "rb") as f:
    HHH_data_3 = pickle.load(f)

with open('/workspaces/premise/premise/carla/carla_samples/HHH_0_50.pkl', "rb") as f:
    HHH_data_4 = pickle.load(f)

with open('/workspaces/premise/premise/carla/carla_samples/HHH_0_50.pkl', "rb") as f:
    HHH_data_5 = pickle.load(f)


#learning_samples = base_data
#learning_samples = base_data +  HHH_data_1
#learning_samples = base_data +  HHH_data_1 + HHH_data_2 
#learning_samples = base_data +  HHH_data_1 + HHH_data_2 + HHH_data_3 
#learning_samples = base_data +  HHH_data_1 + HHH_data_2 + HHH_data_3 + HHH_data_4 
learning_samples = base_data +  HHH_data_1 + HHH_data_2 + HHH_data_3 + HHH_data_4 + HHH_data_5

X = []
y = []

for l in learning_samples:  # 1000 traces 
    #for s in learning_samples[l]:  # s - state
        # Flatten the trace and append to training_data
        flattened_trace = []
        for x in l[:-230]:  # Exclude the last x states (the monitor is given the first (250-x)
            flattened_trace.append(x[1])  # Assuming x[1] contains the feature value (observation)
        X.append(flattened_trace)

        # Determine the prediction based on the last 5 elements in the trace
        if any(x[2] == True for x in l[-230:-180]): # horizon 
            y.append(1)
        else:
            y.append(0)
    

num_steps = 20


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
X_test =  [[('HHH', 'pd60')] * 20]

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


regression_risks = {}

regression_risks['HHH,pd60,20,80'] = [0.03390136,0.03474597,0.03554655,0.03636899,0.03701464,0.03798529]
regression_risks['HHH,pd60,20,50'] = [0.00021963, 0.00023484, 0.00024855, 0.0002198, 0.00021432, 0.00019886]

for r in regression_risks.keys(): 
    print(r, regression_risks)




print(regression_risks)