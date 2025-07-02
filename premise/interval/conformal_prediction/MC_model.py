import numpy as np
import matplotlib.pyplot as plt
import math
import argparse
import pickle


class mc_model():
	
    def __init__(self, horizon):
        self.horizon = horizon

    def gen_trajectories(self, samples, horizon):
        paths = []

        for i in range(len(samples)):
            path = []
            for x in range(len(samples[i]) - horizon):
                val_raw = samples[i][x][0]

                if isinstance(val_raw, (list, tuple, np.ndarray)):
                    vec = np.array(val_raw, dtype=np.float64)
                else:
                    vec = np.array([np.float64(val_raw)])

                path.append(vec)

            paths.append(np.stack(path))  # ensure each path is 2D

        return np.stack(paths)  # returns shape: (batch, time, features)


    def get_noisy_measurments(self, samples, horizon):
        traces = []

        for i in range(len(samples)):
            trace = []
            for x in range(len(samples[i]) - horizon):
                trace.append([np.float64(samples[i][x][1])])  # wrap in list to keep 2D shape
            traces.append(trace)

        return np.array(traces, dtype=np.float64)  # shape: (batch, time, 1)


    def gen_labels(self, samples, horizon):
        labels = []
        for i in range(len(samples)):
            last_labels = [samples[i][x][2] for x in range(len(samples[i]) - horizon, len(samples[i]))]
            label = 1.0 if any(last_labels) else 0.0
            labels.append(label)
        return np.array(labels, dtype=np.float64)



#if __name__=='__main__':
#    samples  = np.load('/workspaces/premise/premise/analysis/test_sets/SnL-10x10_l15_ho5_num200000.npy', allow_pickle=True)
#   print(len(samples))


#    mc_model = mc_model()
#    
#    trajs = mc_model.gen_trajectories(samples,horizon)
#    noisy_measurments = mc_model.get_noisy_measurments(samples,horizon)
#    labels = mc_model.gen_labels(samples,horizon)
	
		
#    print("Percentage of positive points: ", np.sum(labels)/len(samples))
#    dataset_dict = {"x": trajs, "y": noisy_measurments, "cat_labels": labels}

    #print(len(dataset_dict['x']))
    #print(len(dataset_dict['x'][0]))

    #filename = '/workspaces/premise/premise/interval/conformal_prediction/Datasets/SnL_big/SnL_model_validation_set_50.pickle'
    #with open(filename, 'wb') as handle:
    #    pickle.dump(dataset_dict, handle)
    #handle.close()
    #print("Data stored in: ", filename)