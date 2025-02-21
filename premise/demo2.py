from pathlib import Path
import models
import monitor
import trace_generator
import sys

model_def = models.default_models[sys.argv[1] if len(sys.argv) > 1 else "SnL-10x10"]

# Load the model into storm
model, risk = models.build_model_and_risk(
    model_def,
    monitor.PremiseOptions(),
)

states, transtitions = models.build_state_and_transition_list(
    model, model_def.target_label
)
print(states[:10], transtitions[:10])

# Create the conditional trace generator and give it the target label
ctr = trace_generator.ConditionalTraceGenerator(
    model, target_label=model_def.target_label
)

# Generate a random trace not conditioned on anything
path, trace, bad = ctr.generate_random_trace([], 30)

# Print the path
if model.has_state_valuations:
    for state in path:
        print(model.state_valuations.get_string(state))

# Print if the path ended in a target state
print(bad)

# Generate 1000 random traces of length 50 conditioned on the previous trace
res = []
for _ in range(500):
    print(".", end="", flush=True)
    res.append(ctr.generate_random_trace(trace, 50)[2] * 1)

# Show the probability of reaching a bad state after the previous trace within 20 steps
print("\n", sum(res) / len(res))
