import numpy as np
import sklearn
#from sklearn.tree import DecisionTreeClassifier
import sklearn.tree
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LogisticRegression

leaning_samples = np.load('/workspaces/premise/premise/examples/learning_samples.npy')

X = []
y = []


for l in range(len(leaning_samples)):  # one of 25 sets
    for s in leaning_samples[l]:  # 250 sample paths
        
        # Flatten the trace and append to training_data
        flattened_trace = []
        for x in s[:-5]:  # Exclude the last 5 elements (prediction)
            flattened_trace.append(x[1])  # Assuming x[1] contains the feature value
        X.append(flattened_trace)

        # Determine the prediction based on the last 5 elements in the trace
        if any(x[2] == 1 for x in s[-5:]):  # Look at the last 5 steps for error state
            y.append(1)
        else:
            y.append(0)

num_steps = 15
observations = [0, 1, 2, 3]
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

testing_samples = np.load('/workspaces/premise/premise/examples/testing_samples.npy')

X_test = []

for t in testing_samples:
    trace = [x[1] for x in t[0]]  
    X_test.append(trace)


binary_test_data = []
for trace in X_test:
    row = []
    for step in range(num_steps):
        obs = trace[step]  
        row.extend([1 if obs == o else 0 for o in observations])  
    binary_test_data.append(row)


X_test = pd.DataFrame(binary_test_data, columns=column_names)


#risk = []
    #prob = decision_tree.predict_proba([row.values])[0][1]  # Probability of class 1 (error)
prob = model.predict_proba(X_test)
    #decision_tree.predict_proba(pd.DataFrame([row], columns=column_names))[0][1]
    #risk.append(prob)

prob_class_1 = prob[:, 1]

print(len(prob_class_1))

alarms = np.load('/workspaces/premise/premise/examples/alarms.npy')
risk_model_based = np.load('/workspaces/premise/premise/examples/risk_model_based.npy')


error_decision_tree = 0
for x in range(10000): 
    error_decision_tree += (abs(alarms[x] - prob_class_1[x]))

error_model_based = 0 
for x in range(10000): 
    error_model_based += (abs(alarms[x] - risk_model_based[x]))

print(f'Avarage distance decision tree aproach :{error_decision_tree/10000}')
print(f'Avarage distance model based aproach :{error_model_based/10000}')

def error_rates(threashold): 
    not_reached = 0 
    reached = 0 
    false_positve_model_based = 0 
    false_positve_decision_tree = 0
    false_negative_model_based = 0  
    false_negative_decision_tree = 0 

    for x in range(10000):

        if alarms[x] == 0:
            not_reached += 1 
            if (risk_model_based[x] > (threashold)):
                false_positve_model_based += 1
            if (prob_class_1[x] > (threashold)):
                false_positve_decision_tree += 1

        if alarms[x] == 1:
            reached += 1 
            if (risk_model_based[x] < (threashold)): 
                false_negative_model_based +=1
            if (prob_class_1[x] < (threashold)): 
                false_negative_decision_tree +=1 
    
    return(false_positve_model_based/not_reached, false_positve_decision_tree/not_reached, false_negative_model_based/reached, false_negative_decision_tree/reached, (false_positve_model_based+false_negative_model_based)/10000, (false_positve_decision_tree+false_negative_decision_tree)/10000)


for_graph = []

for x in range(10,500): 
    for_graph.append(error_rates(x/1000))


x = []


for a in range(10, 500): 
    x.append(a/1000)

y0 = []
y1 = []
y2 = []
y3 = []
y4 = []
y5 = []


for a in for_graph: 
    y0.append(a[0])
    y1.append(a[1])
    y2.append(a[2])
    y3.append(a[3])
    y4.append(a[4])
    y5.append(a[5])


y6 = []
y7 = []


for b in for_graph: 
    y6.append(b[0] + b[2]) 
    y7.append(b[1] + b[3])



plt.plot(x, y0, label = "Model Based approach false positive")
plt.plot(x, y1, label = "Decison Tree approach false positive")

plt.xlabel("Threashold")
plt.ylabel("False Positive raten")
plt.legend(title = "Legend")
plt.title("Change in rate of false positives")

plt.show(block=True)

plt.plot(x, y2, label = "Model Based approach false negative")
plt.plot(x, y3, label = "Decison Tree approach false negative")

plt.xlabel("Threashold")
plt.ylabel("False Negative rate")
plt.legend(title = "Legend")
plt.title("Change in rate of false negatives")

plt.show(block=True) 

plt.plot(x, y4, label = "Model Based approach")
plt.plot(x, y5, label = "Decison Tree approach")

plt.xlabel("Threashold")
plt.ylabel("Error rate")
plt.legend(title = "Legend")
plt.title("Change in rate of error")

plt.show(block=True)

plt.plot(x, y6, label = "Model Based approach")
plt.plot(x, y7, label = "Decison Tree approach")

plt.xlabel("Threashold")
plt.ylabel("False negative + False postive")
plt.legend(title = "Legend")
plt.title("Change in rate False negative + False postive")

plt.show(block=True)

print("FLASE NEGATIVE BELOW 1/5")


accceptable_mb = []
for x in range(len(y0)): 
    if y2[x] < (1/5):
        accceptable_mb.append(x)
        
    
acceptable_mf = [] 
for x in range(len(y1)): 
    if y3[x] < (1/5):
        acceptable_mf.append(x)


plot3 = []
for x in accceptable_mb: 
    plot3.append(y4[x])

plot4 = []
for x in acceptable_mf: 
    plot4.append(y5[x])  


plt.plot(accceptable_mb, plot3, label = "Model Based approach ")
plt.plot(acceptable_mf, plot4, label = "Decison Tree approach ")

plt.xlabel("Threashold")
plt.ylabel("Error rate")
plt.legend(title = "Legend")
plt.title("Change in rate of error")

plt.show(block=True) 


plot5 = []
for x in accceptable_mb: 
    plot5.append(y6[x])

plot6 = []
for x in acceptable_mf: 
    plot6.append(y7[x])  


plt.plot(accceptable_mb, plot5, label = "Model Based approach ")
plt.plot(acceptable_mf, plot6, label = "Decison Tree approach ")

plt.xlabel("Threashold")
plt.legend(title = "Legend")
plt.title("False negative + Flase Positive")

plt.show(block=True) 


print('____________________________________________________')
error = []
false_p = []

for x in range(len(y0)): 
    if y2[x] < (1/10):
        false_p.append(y0[x])
        error.append(y4[x])

if false_p and error:
    min_false_p = min(false_p)
    min_error = min(error)

    optimal_fp = []
    optimal_e = []

    print("Optimal model-based approach performance, false negative rate below 1/10")

    for x in range(len(y0)): 
        if y0[x] == min_false_p: 
            optimal_fp.append(x)

    for x in range(len(y0)): 
        if y4[x] == min_error: 
            optimal_e.append(x)

    for index in optimal_fp:
        if index in optimal_e:
            print(index, f'false positive: {y0[index]}', f'false negative: {y2[index]}', 
                  f'error rate: {y4[index]}', f'false positive and false negative: {y0[index] + y2[index]}')

print('____________________________________________________')

error_dc = []
false_p_dc = []

for x in range(len(y1)): 
    if y3[x] < (1/10):
        false_p_dc.append(y1[x])
        error_dc.append(y5[x])

if false_p_dc and error_dc:
    min_false_p_dc = min(false_p_dc)
    min_error_dc = min(error_dc)

    print("Optimal model-free performance, false negative rate below 1/10")

    optimal_dc_fp = []
    optimal_dc_e = []

    for x in range(len(y1)): 
        if y1[x] == min_false_p_dc:
            optimal_dc_fp.append(x)

    for x in range(len(y1)):
        if y5[x] == min_error_dc:
            optimal_dc_e.append(x)

    for index in optimal_dc_fp:
        if index in optimal_dc_e:
            print(index, f'false positive: {y1[index]}', f'false negative: {y3[index]}', 
                  f'error rate: {y5[index]}', f'false positive and false negative: {y1[index] + y3[index]}')


