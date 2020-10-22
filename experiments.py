import stormpy as sp
import click

import demo

class Benchmark:
    """
    Container for benchmarks.
    """

    def __init__(self, name, modelpath, constants, risk_def):
        self.name = name
        self.modelpath = modelpath
        self.constants = constants
        self.risk_def = risk_def

# Benchmarks
benchmarks = [
#    Benchmark("airportA-3-50-30", "examples/airportA-3.nm", "DMAX=50,PMAX=30", "Pmax=? [F \"crash\"]"),
    Benchmark("airportA-7-50-30", "examples/airportA-7.nm", "DMAX=50,PMAX=30", "Pmax=? [F \"crash\"]"),
    Benchmark("airportB-3-50-30", "examples/airportB-3.nm", "DMAX=50,PMAX=30", "Pmax=? [F \"crash\"]"),
    Benchmark("airportB-7-50-30", "examples/airportB-7.nm", "DMAX=50,PMAX=30", "Pmax=? [F \"crash\"]"),
#    Benchmark("evadeI-10", "examples/hidden-incentive.nm", "N=10", "Pmax=? [F<=12 \"crash\"]"),
    Benchmark("evadeI-15", "examples/hidden-incentive.nm", "N=15", "Pmax=? [F<=12 \"crash\"]"),
    Benchmark("evadeV-5-3", "examples/evade-monitoring.nm", "N=5,RADIUS=3", "Pmax=? [F<=12 \"crash\"]"),
    Benchmark("evadeV-6-3", "examples/evade-monitoring.nm", "N=6,RADIUS=3", "Pmax=? [F<=12 \"crash\"]"),
    Benchmark("refuelA-12-50", "examples/refuel.nm", "N=12,ENERGY=50", "Pmax=? [F<=12 \"empty\"]"),
    Benchmark("refuelB-12-50","examples/refuelB.nm", "N=12,ENERGY=50", "Pmax=? [F<=12 \"empty\"]")
]

if __name__ == "__main__":
    # Wait for termination, never crash.
    sp.set_settings(["--signal-timeout", "100000"])


    nr_traces = 50
    trace_length = 500
    promtness_deadline = 1000 # in ms
    configurations = [demo.UnfoldingOptions(exact_arithmetic=True),
                      demo.UnfoldingOptions(exact_arithmetic=False),
                      demo.ForwardFilteringOptions(exact_arithmetic=True, convex_hull_reduction=False),
                      demo.ForwardFilteringOptions(exact_arithmetic=True, convex_hull_reduction=True)]
    for benchmark in benchmarks:
        for config in configurations:
            try:
                demo.monitor(benchmark.modelpath, benchmark.risk_def, benchmark.constants, trace_length, config, verbose=False, promptness_deadline=promtness_deadline, simulator_seed=range(nr_traces), model_id=benchmark.name)
            except RuntimeWarning:
                pass