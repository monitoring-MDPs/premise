import argparse
from numbers import Number
from typing import Any
import numpy as np
import stormpy.pomdp
import tqdm

from interval import create_monitor
import models
import monitor
import trace_generator
import matplotlib.pyplot as plt
import pickle


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Conformance checking")
    parser.add_argument(
        "-m",
        "--model_name",
        type=str,
        default="SnL-10x10",
        help="Name of the model to use",
    )
    parser.add_argument(
        "-t", "--trans_path", type=str, help="Path to the transition dictionary"
    )
    parser.add_argument(
        "-i", "--init_path", type=str, help="Path to the initial state dictionary"
    )
    parser.add_argument(
        "-l", "--sample_length", type=int, help="Length of the samples to generate"
    )
    parser.add_argument(
        "--conformence-amount",
        type=int,
        default=1000,
        help="Amount of extensions to generate for a sample",
    )
    parser.add_argument("--horizon", type=int, help="The horizon to monitor on")
    parser.add_argument(
        "--dump", type=str, help="Path to the file to dump the model to"
    )
    parser.add_argument("--verbose", "-v", action="count", default=0)

    args = parser.parse_args()

    
    refinement_results = {}
    x = []
    y1 = []
    y2 = []
    y3 = []

    tested_samples = []

    # Get the real model definition
    model_def = models.default_models[args.model_name]
    model_def.risk_property = f'Pmax=? [F<={args.horizon} "{model_def.target_label}" ]'

    # Load the real model into storm
    model, real_risk = models.build_model_and_risk(
            model_def,
            monitor.PremiseOptions(),
        )

    risks = []
    sample_risks = []
    new_sample_index = [0]


    for i in range(50): 

        if i == 0:
            trans_path = "/workspaces/premise/premise/examples/SnL-10x10-interval.npy"
            init_path = "/workspaces/premise/premise/examples/SnL-10x10-initial_interval.npy"
        else:
            trans_path = f"/workspaces/premise/premise/examples/SnL-10x10_{i+1}-interval.npy"
            init_path = f"/workspaces/premise/premise/examples/SnL-10x10_{i+1}-initial_interval.npy"


        # Load learned model
        interval = np.load(trans_path, allow_pickle=True)[()]
        initial_interval = np.load(init_path, allow_pickle=True)[()]

        ctr = trace_generator.ConditionalTraceGenerator(
            model, target_label=model_def.target_label
        )

        if i == 0: 
            sample = [ctr.generate_random_trace([], args.sample_length)]

        print(f"Running iteration {i+1} with:")
        print("Ready for conformance checking")

    
        # Build the premise monitor on the learned model
        mon, observation_map, unfolder, ipomdp = create_monitor(
            interval,
            initial_interval,
            "min",
            True,
            args.horizon,
            args.dump,
            args.verbose,
        )  
        print(trans_path)
        print(init_path)


        for trace in tqdm.tqdm(sample):
            mon.initialize(0)
            observations = [t[1] for t in trace]
            for obs in observations[:-1]:
                mon.step(observation_map[obs], compute_risk=False)
            last_risk = mon.step(observation_map[observations[-1]], compute_risk=True)
            risk = (trace, last_risk)

        risks.append(risk[1])

        alarm = 0
        for _ in range(args.conformence_amount):
            new_trace = ctr.generate_random_trace(
                [s[1] for s in trace], length=len(trace) + args.horizon
            )
            alarm += any([s[2] for s in new_trace])

        sample_risks.append((alarm/args.conformence_amount)) 

        if i > 5 and ((i - 5) >= new_sample_index[-1]): 
            if all(abs((sample_risks[x]) - risks[x]) < 0.05 for x in range(i-5,i+1)):
                sample = [ctr.generate_random_trace([], args.sample_length)]
                new_sample_index.append(i)
                
                
    with open('/workspaces/premise/premise/examples/refinement_per_trace_results.pkl', 'wb') as f:
        pickle.dump(risks, f)

    with open('/workspaces/premise/premise/examples/refinement_per_trace_sample_results.pkl', 'wb') as f:
        pickle.dump(sample_risks, f)
    
    with open('/workspaces/premise/premise/examples/refinement_per_trace_new_sample_index.pkl', 'wb') as f:
        pickle.dump(new_sample_index, f)
    
  



 #/opt/venv/bin/python /workspaces/premise/premise/refinement_per_trace.py -i /workspaces/premise/premise/examples/SnL-10x10-initial_interval.npy -t /workspaces/premise/premise/examples/SnL-10x10-interval.npy -l 10 --horizon 5    
    
    
