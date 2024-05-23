import stormpy as sp
import argparse
import monitoring

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

challenges = [
   # Benchmark("airportA-7-400-40", "examples/airportA-7.nm", "DMAX=400,PMAX=40", "Pmax=? [F \"crash\"]"),
   # Benchmark("airportB-3-200-30", "examples/airportB-3.nm", "DMAX=200,PMAX=30", "Pmax=? [F \"crash\"]"),
   # Benchmark("airportB-7-200-30", "examples/airportB-7.nm", "DMAX=200,PMAX=30", "Pmax=? [F \"crash\"]"),
#    Benchmark("evadeI-10", "examples/hidden-incentive.nm", "N=10", "Pmax=? [F<=12 \"crash\"]"),
   #  Benchmark("evadeI-19", "examples/hidden-incentive.nm", "N=19", "Pmax=? [F<=20 \"crash\"]"),
#     Benchmark("evadeV-5-3", "examples/evade-monitoring.nm", "N=5,RADIUS=3", "Pmax=? [F<=12 \"crash\"]"),
 #    Benchmark("evadeV-9-3", "examples/evade-monitoring.nm", "N=9,RADIUS=3", "Pmax=? [F<=12 \"crash\"]"),
    #Benchmark("evadeV-14-4", "examples/evade-monitoring.nm", "N=14,RADIUS=4", "Pmax=? [F<=12 \"crash\"]"),
    Benchmark("refuelA-35-80", "examples/refuel.nm", "N=35,ENERGY=80", "Pmax=? [F<=20 \"empty\"]"),
  #   Benchmark("refuelB-14-200","examples/refuelB.nm", "N=18,ENERGY=200", "Pmax=? [F<=8 \"empty\"]")
]

environment = sp.Environment()
#environment.solver_environment.minmax_solver_environment.method = sp.MinMaxMethod.linear_programming
environment.solver_environment.minmax_solver_environment.precision = sp.Rational("0.01")

configurations = [#monitoring.UnfoldingOptions(environment, exact_arithmetic=True),
                  monitoring.UnfoldingOptions(environment, exact_arithmetic=True, custom_str="unfrefactored")]
                  #monitoring.ForwardFilteringOptions(exact_arithmetic=True, convex_hull_reduction=False),
                  #monitoring.ForwardFilteringOptions(exact_arithmetic=True, convex_hull_reduction=True)]

if __name__ == "__main__":
    # Wait for termination, never crash.
    sp.set_settings(["--signal-timeout", "100000"])
    parser = argparse.ArgumentParser(description="Run experiments with premise.")
    parser.add_argument("--number-traces", default=10, type=int, help="How many traces to run")
    parser.add_argument("--trace-length", default=100, type=int, help="How long should the traces be?")
    parser.add_argument("--promptness-deadline", default=1000, type=int, help="How long may one iteration take at most?")
    parser.add_argument("--verbose", action='store_true', help="Enable extra output")
    args = parser.parse_args()

    nr_traces = args.number_traces
    trace_length = args.trace_length
    promtness_deadline = args.promptness_deadline # in ms
    for benchmark in benchmarks:
        for config in configurations:
            print(f"Running {benchmark.name} with {str(config)}")
            try:
                monitoring.run_monitor(benchmark.modelpath, benchmark.risk_def, benchmark.constants, trace_length, config, verbose=args.verbose, promptness_deadline=promtness_deadline, simulator_seed=range(nr_traces), model_id=benchmark.name)
            except RuntimeWarning:
                print("Skipped (likely, the folder exists)")
                pass