import models
import monitor
import trace_generator
import traces
import oracle

def construct_learning_interfaces(model_description : models.ModelDescription,
                                  premise_options : monitor.PremiseOptions) \
        -> (oracle.Oracle, trace_generator.SimulationTraceGenerator):
    model, risk_structure = models.build_model_and_risk(model_description, premise_options)
    mon = monitor.initialize_monitor(model, risk_structure, premise_options)
    tracegen = trace_generator.make_simulation_wrapper(model, risk_structure)
    tracemapper = traces.TraceMapper(model)
    return oracle.Oracle(mon), tracegen, tracemapper

oracle_interface, generator, tracemapper = construct_learning_interfaces(models.default_models["evadeV-5-3"], monitor.PremiseOptions())
trace = generator.generate_random_trace(20)
hl_trace = tracemapper.trace_to_high_level(trace)
assert tracemapper.trace_from_high_level_trace(hl_trace) == trace
risk = oracle_interface.membership(trace, intermediate_results=True)
assert len(risk) == len(trace)
print(risk)
