from pathlib import Path
import models
import monitor
import trace_generator

model, risk = models.build_model_and_risk(
    models.ModelDescription(
        Path(__file__).parent / "examples/airportA-7.nm",
        "DMAX=5,PMAX=5",
        'Pmax=? [F "crash"]',
    ),
    monitor.PremiseOptions(),
)
ctr = trace_generator.ConditionalTraceGenerator(model, "crash")
path, trace, bad = ctr.generate_random_trace([], 30)
if model.has_state_valuations:
    for state in path:
        print(model.state_valuations.get_string(state))
print(bad)
res = []
for _ in range(1000):
    print(".", end="")
    res.append(ctr.generate_random_trace(trace, 50)[2] * 1)
print("\n", sum(res) / len(res))
