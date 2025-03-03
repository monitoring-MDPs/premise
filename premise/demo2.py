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
sample = ctr.generate_random_trace([], 20)

# Print if the path ended in a target state
print(sample)

# Print the path
if model.has_state_valuations:
    for s in sample:
        print(model.state_valuations.get_string(s[0]), s[1], end=" -> ")
    print()

sample = [
    ("[pos=0]", 3, False),
    ("[pos=5]", 0, False),
    ("[pos=9]", 2, False),
    ("[pos=31]", 0, False),
    ("[pos=34]", 0, False),
    ("[pos=38]", 0, False),
    ("[pos=44]", 2, False),
    ("[pos=49]", 0, False),
    ("[pos=11]", 0, False),
    ("[pos=15]", 0, False),
]

# Generate 500 random traces of length 50 conditioned on the previous trace
res = []
for _ in range(500):
    trace = ctr.generate_random_trace([s[1] for s in sample], length=20)
    res.append(trace[-1][2])
    # if model.has_state_valuations:
    #     for s in trace:
    #         if s[2]:
    #             print("\033[92m", end="")
    #         print(model.state_valuations.get_string(s[0]), s[1], end=" -> ")
    #         if s[2]:
    #             print("\033[0m", end="")
    #     print()


# Show the probability of reaching a bad state after the previous trace within 20 steps
print("\n", sum(res) / len(res))
