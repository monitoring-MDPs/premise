import pickle
import numpy as np 


file_0_100 = "/workspaces/premise/premise/results/testdata_rq_1_airportB-7-40-20_coarse_0_100_every_5th.pkl"
#file_100_200 = ""
file_50 = "/workspaces/premise/premise/results/testdata_rq_1_airportB-7-40-20_coarse_50_every_5th.pkl"

with open(file_0_100, "rb") as f:
    data_0_100 = pickle.load(f)


#with open(file_100_200, "rb") as f:
#    data_100_200 = pickle.load(f)


with open(file_50, "rb") as f:
    data_50 = pickle.load(f)  


print(data_0_100.keys())


data = {}

data['model'] = data_0_100['model']
data['coarse'] = data_0_100['coarse']
data['high_st'] = data_0_100['high_st']
data['mc_transition_counts'] = data_0_100['mc_transition_counts']
data['imc_transition_counts'] = data_0_100['imc_transition_counts']
data['imc_transition_counts_ref'] = data_0_100['imc_transition_counts_ref']
data['imc_transition_counts_refsplit'] = data_0_100['imc_transition_counts_refsplit']


data['testing_samples'] = data_0_100['testing_samples'] + data_50['testing_samples']
data['testing_samples_amount']  = len(data['testing_samples'])

print(data['testing_samples_amount'])

data['alarms'] = np.concatenate([
    data_0_100['alarms'],
    #data_100_200['alarms'],
    data_50['alarms']
])

data['target_risks'] = np.concatenate([
    data_0_100['target_risks'],
    #data_100_200['target_risks'],
    data_50['target_risks']
])


data['mc_risks'] = {k: data_0_100['mc_risks'].get(k, []) + data_50['mc_risks'].get(k, []) for k in set(data_0_100['mc_risks']) | set(data_50['mc_risks'])}

#combined = {
#    str(key): data_0_100['imc_risks'][key] + data_100_200['imc_risks'][key]
#    for key in data_0_100['imc_risks']
#}

data['imc_risks'] = {k: data_0_100['imc_risks'].get(k, []) + data_50['imc_risks'].get(k, []) for k in set(data_0_100['imc_risks']) | set(data_50['imc_risks'])}


#combined = {
#    str(key): data_0_100['imc_risks_ref'][key] + data_100_200['imc_risks_ref'][key]
#    for key in data_0_100['imc_risks_ref']
#}

data['imc_risks_ref'] = {k: data_0_100['imc_risks_ref'].get(k, []) + data_50['imc_risks_ref'].get(k, []) for k in set(data_0_100['imc_risks_ref']) | set(data_50['imc_risks_ref'])}


#combined = {
#    str(key): data_0_100['imc_risks_ref_splitting'][key] + data_100_200['imc_risks_ref_splitting'][key]
#   for key in data_0_100['imc_risks_ref_splitting']
#}

data['imc_risks_refsplit'] = {k:  data_0_100['imc_risks_refsplit'].get(k, []) + data_50['imc_risks_refsplit'].get(k, []) for k in set(data_0_100['imc_risks_refsplit']) | set(data_50['imc_risks_refsplit'])}



with open('/workspaces/premise/premise/results/testdata_rq_1_airportB-7-40-20_coarse_combined.pkl', 'wb') as f:
    pickle.dump(data, f)