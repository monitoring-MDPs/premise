import premise.models as models
import premise.monitor as monitor
import premise.trace_generator as trace_generator
import premise.traces as traces
import premise.learning.oracle as oracle

def construct_learning_interfaces(model_description : models.ModelDescription,
                                  premise_options : monitor.PremiseOptions = monitor.PremiseOptions()) \
        -> (oracle.Oracle, trace_generator.SimulationTraceGenerator):
    model, risk_structure = models.build_model_and_risk(model_description, premise_options)
    mon = monitor.initialize_monitor(model, risk_structure, premise_options)
    tracegen = trace_generator.make_simulation_wrapper(model, risk_structure, premise_options.simulator_seed)
    tracemapper = traces.TraceMapper(model)
    return oracle.Oracle(mon), tracegen, tracemapper
