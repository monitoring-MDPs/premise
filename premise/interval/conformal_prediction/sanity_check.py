
import numpy as np
from MC_model import mc_model

model = mc_model()
horizon = 5 

#samples_for_testing = np.load('/Users/skurka/premise/premise/analysis/test_sets/SnL-10x10_l15_ho5.npy', allow_pickle=True)
#samples_for_testing = samples_for_testing[9900:]
#noisy_measurements = model.gen_labels(samples_for_testing, horizon)

#print(noisy_measurements)


#samples = np.load('/workspaces/premise/premise/analysis/test_sets/SnL-10x10_l15_ho5_num200000.npy', allow_pickle=True)

#print(samples[0])

#active_samples = samples[62050:63050]
#np.save('/workspaces/premise/premise/interval/conformal_prediction/Datasets/SnL_big/SnL_model_final_test_1K.npy', active_samples)




  

	