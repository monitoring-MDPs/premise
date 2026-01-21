import premise
import models
import monitor
import trace_generator
import traces
import learning.oracle

oracle_interface, generator, tracemapper = premise.construct_learning_interfaces(models.default_models["evadeV-5-3"], monitor.PremiseOptions())
trace = generator.generate_random_trace(20)
hl_trace = tracemapper.trace_to_high_level(trace)
assert tracemapper.trace_from_high_level_trace(hl_trace) == trace
risk = oracle_interface.membership(trace, intermediate_results=True)
assert len(risk) == len(trace)
print(risk)
