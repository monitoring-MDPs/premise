import argparse

import stormpy

import monitoring
import random


def main():
    parser = argparse.ArgumentParser(description="Run premise simulating from model.")
    methodgroup = parser.add_mutually_exclusive_group(required=True)
    methodgroup.add_argument("--filtering", action="store_true")
    methodgroup.add_argument("--unfolding", action="store_true")
    numericsgroup = parser.add_mutually_exclusive_group(required=True)
    numericsgroup.add_argument("--exact", action="store_true")
    numericsgroup.add_argument("--float", action="store_false", dest="exact")
    parser.add_argument("--name", default="default_name", help="Name for stat output")
    parser.add_argument("--model", required=True, help="Path to models")
    parser.add_argument(
        "--constants", default="", type=str, help="Constant definition string"
    )
    parser.add_argument(
        "--risk", required=True, help="Property defining the state risk"
    )
    parser.add_argument(
        "--number-traces", default=50, type=int, help="How many traces to run"
    )
    parser.add_argument(
        "--trace-length", default=500, type=int, help="How long should the traces be?"
    )
    parser.add_argument(
        "--promptness-deadline",
        default=1000,
        type=int,
        help="How long may one iteration take at most?",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable extra output")
    parser.add_argument("--seed", help="Set a random seed for reproducible experiments")
    parser.add_argument(
        "--no-convexhull-reduction",
        help="Disable Convexhull Reduction",
        dest="convexhull",
        action="store_false",
    )
    parser.add_argument(
        "--dump-models",
        type=str,
        help="Path for dumping models. If none given, no models are dumped.",
    )
    parser.add_argument(
        "--unfolding-mode",
        choices=["rejection_sampling", "storm-conditional"],
        default="rejection_sampling",
    )
    parser.add_argument(
        "--conditional-method",
        choices=["default", "restart", "bisection", "bisection-advanced", "pi"],
        default="default",
    )
    parser.add_argument(
        "--mc-method",
        choices=[
            "policy_iteration",
            "value_iteration",
            "sound_value_iteration",
            "optimistic_value_iteration",
        ],
    )
    args = parser.parse_args()

    trace_length = args.trace_length
    promptness_deadline = args.promptness_deadline  # in ms
    if args.seed is None:
        seed = [random.getrandbits(64) for _ in range(args.number_traces)]
    else:
        random.seed(args.seed)
        seed = [random.getrandbits(64) for _ in range(args.number_traces)]

    if args.filtering:
        options = monitoring.ForwardFilteringOptions(
            exact_arithmetic=args.exact, convex_hull_reduction=args.convexhull
        )
    elif args.unfolding:
        options = monitoring.UnfoldingOptions(
            exact_arithmetic=args.exact,
            export_models_path=args.dump_models,
            use_rejection_sampling=args.unfolding_mode == "rejection_sampling",
            conditional_method=args.conditional_method,
            model_checking_method=args.mc_method,
        )
    else:
        raise RuntimeError("Unknown method!")

    if args.verbose:
        import os

        input(os.getpid())
        stormpy.set_loglevel_trace()

    monitoring.run_monitor(
        args.model,
        args.risk,
        args.constants,
        trace_length,
        options,
        args.verbose,
        seed,
        promptness_deadline,
        args.name,
    )


if __name__ == "__main__":
    main()
