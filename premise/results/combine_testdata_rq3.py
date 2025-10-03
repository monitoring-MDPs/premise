import pickle
import numpy as np 


file_0_100 = "/workspaces/premise/premise/results/testdata_rq_3_airportA-7-40-20_high_st_0_100_every.pkl"
file_100_200 = "/workspaces/premise/premise/results/testdata_rq_3_airportA-7-40-20_high_st_100_200_every.pkl"
file_50 = "/workspaces/premise/premise/results/testdata_rq_3_airportA-7-40-20_high_st_50_every.pkl"

with open(file_0_100, "rb") as f:
    data_0_100 = pickle.load(f)


with open(file_100_200, "rb") as f:
    data_100_200 = pickle.load(f)


with open(file_50, "rb") as f:
    data_50 = pickle.load(f)  


print(data_0_100.keys())


data = {}

data['model'] = data_0_100['model']
data['coarse'] = data_0_100['coarse']
data['high_st'] = data_0_100['high_st']
data['stopping_threshold'] = data_0_100['stopping_threshold']
data['horizon'] = data_0_100['horizon']
data['initial_amount'] = data_0_100['initial_amount']
data['imc_transition_counts'] = data_0_100['imc_transition_counts']
data['imc_transition_counts_ref'] = data_0_100['imc_transition_counts_ref']
data['imc_transition_counts_ref_splitting'] = data_0_100['imc_transition_counts_ref_splitting']

data['testing_samples'] = data_0_100['testing_samples'] + data_100_200['testing_samples'] + data_50['testing_samples']


data['alarms'] = np.concatenate([
    data_0_100['alarms'],
    data_100_200['alarms'],
    data_50['alarms']
])

data['target_risks'] = np.concatenate([
    data_0_100['target_risks'],
    data_100_200['target_risks'],
    data_50['target_risks']
])

combined = {
    str(key): data_0_100['imc_risks'][key] + data_100_200['imc_risks'][key]
    for key in data_0_100['imc_risks']
}
data['imc_risks'] = {k: combined.get(k, []) + data_50['imc_risks'].get(k, []) for k in set(combined) | set(data_50['imc_risks'])}


combined = {
    str(key): data_0_100['imc_risks_ref'][key] + data_100_200['imc_risks_ref'][key]
    for key in data_0_100['imc_risks_ref']
}
data['imc_risks_ref'] = {k: combined.get(k, []) + data_50['imc_risks_ref'].get(k, []) for k in set(combined) | set(data_50['imc_risks_ref'])}


combined = {
    str(key): data_0_100['imc_risks_ref_splitting'][key] + data_100_200['imc_risks_ref_splitting'][key]
    for key in data_0_100['imc_risks_ref_splitting']
}
data['imc_risks_ref_splitting'] = {k: combined.get(k, []) + data_50['imc_risks_ref_splitting'].get(k, []) for k in set(combined) | set(data_50['imc_risks_ref_splitting'])}


combined = {
    str(key): data_0_100['imc_distances'][key] + data_100_200['imc_distances'][key]
    for key in data_0_100['imc_distances']
}
data['imc_distances'] = {k: combined.get(k, []) + data_50['imc_distances'].get(k, []) for k in set(combined) | set(data_50['imc_distances'])}


combined = {
    str(key): data_0_100['imc_distances_ref'][key] + data_100_200['imc_distances_ref'][key]
    for key in data_0_100['imc_distances_ref']
}
data['imc_distances_ref'] = {k: combined.get(k, []) + data_50['imc_distances_ref'].get(k, []) for k in set(combined) | set(data_50['imc_distances_ref'])}



combined = {
    str(key): data_0_100['imc_distances_ref_splitting'][key] + data_100_200['imc_distances_ref_splitting'][key]
    for key in data_0_100['imc_distances_ref_splitting']
}
data['imc_distances_ref_splitting'] = {k: combined.get(k, []) + data_50['imc_distances_ref_splitting'].get(k, []) for k in set(combined) | set(data_50['imc_distances_ref_splitting'])}


combined = {
    str(key): np.concatenate([
        data_0_100['regression_risks'][key],
        data_100_200['regression_risks'][key]
    ])
    for key in data_0_100['regression_risks']
}

data['regression_risks'] = {
    str(k): np.concatenate([
        combined.get(k, []),
        data_50['regression_risks'].get(k, [])
    ])
    for k in set(combined) | set(data_50['regression_risks'])
}


combined = {
    str(key): data_0_100['conformal_risks'][key] + data_100_200['conformal_risks'][key]
    for key in data_0_100['conformal_risks']
}
data['conformal_risks'] = {str(k): combined.get(k, []) + data_50['conformal_risks'].get(k, []) for k in set(combined) | set(data_50['conformal_risks'])}


with open('/workspaces/premise/premise/results/testdata_rq_3_airportA-7-40-20_high_st_combined.pkl', 'wb') as f:
    pickle.dump(data, f)