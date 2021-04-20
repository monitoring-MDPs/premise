import argparse
import monitoring
import random

def main():
    parser = argparse.ArgumentParser(description="Run premise simulating from model.")
    methodgroup = parser.add_mutually_exclusive_group(required=True)
    methodgroup.add_argument('--filtering', action='store_true')
    methodgroup.add_argument('--unfolding', action='store_true')
    numericsgroup = parser.add_mutually_exclusive_group(required=True)
    numericsgroup.add_argument('--exact', action='store_true')
    numericsgroup.add_argument('--float', action='store_false', dest="exact")
    parser.add_argument("--name", default="default_name", help="Name for stat output")
    parser.add_argument("--model", required=True, help="Path to models")
    parser.add_argument("--constants", default="", type=str, help="Constant definition string")
    parser.add_argument("--risk", required=True, help="Property defining the state risk")
    parser.add_argument("--number-traces", default=50, type=int, help="How many traces to run")
    parser.add_argument("--trace-length", default=500, type=int, help="How long should the traces be?")
    parser.add_argument("--promptness-deadline", default=1000, type=int, help="How long may one iteration take at most?")
    parser.add_argument("--verbose", action='store_true', help="Enable extra output")
    parser.add_argument("--seed", help="Set a random seed for reproducible experiments")
    parser.add_argument("--no-convexhull-reduction", help="Disable Convexhull Reduction", dest="convexhull", action='store_false')
    args = parser.parse_args()

    trace_length = args.trace_length
    promptness_deadline = args.promptness_deadline  # in ms
    if args.seed is None:
        seed = [random.getrandbits(64) for _ in range(args.number_traces)]
    else:
        random.seed(args.seed)
        seed = [random.getrandbits(64) for _ in range(args.number_traces)]

    if args.filtering:
        options = monitoring.ForwardFilteringOptions(exact_arithmetic=args.exact, convex_hull_reduction=args.convexhull)
    elif args.unfolding:
        options = monitoring.UnfoldingOptions(exact_arithmetic=args.exact)
    else:
        RuntimeError("Unknown method!")

    monitoring.monitor(args.model, args.risk, args.constants, trace_length, options, args.verbose, seed, promptness_deadline, args.name)

if __name__ == "__main__":
    main()