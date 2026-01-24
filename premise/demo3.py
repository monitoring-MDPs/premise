import time
from pathlib import Path


import premise
import models
import monitor
import trace_generator
import traces
import learning.oracle as oracle

def ensure_path_exists(path):
    Path(path).mkdir(parents=True, exist_ok=True)

options = monitor.PremiseOptions()
options.rejection_sampling = True
options.promptness_deadline = 1000000000

# "evadeV-5-3", "evadeV-12-4","evadeV-14-4", "evadeV-9-3",
#"evadeI-19", "refuelB-18-200",
# "patrol-12-2", "patrol-15-3",
#modelnames = ["refuelA-12-50", "refuelC-12-40", "refuelC-25-20"]
modelnames = ["evadeIG-15-4"]

for modelname in modelnames:
    oracle_interface, generator, tracemapper = premise.construct_learning_interfaces(models.default_models[modelname],
                                                                                     options)
    print("Model created...")

    for seed in range(20):
        for trlen in [25,50,75,100,150,200,250,300,350,400,600,1000]:
            generator.set_seed(seed)
            trace = generator.generate_random_trace(trlen)
            start_time = time.time()
            risk = oracle_interface.membership(trace, intermediate_results=False)
            end_time = time.time()
            if risk is not None:
                print(float(risk))
            else:
                print("TO")
            print(end_time - start_time)
            total_time = end_time - start_time
            if risk is None or (float(risk) > 0.001 and risk < 1 and total_time > 0.1):
                path = f"filedump-cd/{modelname}-{seed}"
                oracle_interface.dump_internal_data(f"{path}")
