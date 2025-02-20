from pathlib import Path
import models
import monitor
import trace_generator

# Load the model into storm
model, risk = models.build_model_and_risk(
    models.ModelDescription(
        Path(__file__).parent / "examples/airportA-7.nm",
        "DMAX=5,PMAX=5",
        'Pmax=? [F "crash"]',
    ),
    monitor.PremiseOptions(),
)

# Create the conditional trace generator and give it the target label
ctr = trace_generator.ConditionalTraceGenerator(model, "crash")

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
for _ in range(1000):
    print(".", end="")
    res.append(ctr.generate_random_trace(trace, 50)[2] * 1)

# Show the probability of reaching a bad state after the previous trace within 20 steps
print("\n", sum(res) / len(res))
