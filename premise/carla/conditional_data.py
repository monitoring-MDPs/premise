import pickle

all_data = [] 


with open("/workspaces/premise/premise/carla/carla_conditional_samples/HLL/batch_0.pkl", "rb") as f:
    data1 = pickle.load(f)
    for x in data1: 
        all_data.append(x)

print(len(all_data)) 