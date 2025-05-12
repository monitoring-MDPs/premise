import signal
import sys
from typing import NoReturn

from premise.interval.regression_model import reg_argsparser, reg_main
from premise.interval.refinement import ref_args_parser, ref_main


def split_args(args, delim):
    res = [[]]
    for a in args:
        if a == delim:
            res.append([])
        else:
            res[-1].append(a)
    return res


def timeout_handler(signum, frame) -> NoReturn:
    print("Timeout occurred")
    exit(2)
    raise Exception("Timeout occurred")


if __name__ == "__main__":
    args = split_args(sys.argv[3:], "::")

    # Calculate timeout
    if sys.argv[1][-1] == "h":
        timeout = int(sys.argv[1][:-1]) * 60 * 60
    elif sys.argv[1][-1] == "m":
        timeout = int(sys.argv[1][:-1]) * 60
    elif sys.argv[1][-1] == "s":
        timeout = int(sys.argv[1][:-1])

    print("Timeout: ", timeout)

    if sys.argv[2] == "refinement":
        ref_parser = ref_args_parser()
        ref_args = ref_parser.parse_args(args[0])
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(timeout)
        try:
            ref_main(ref_args)
        except TimeoutError:
            print("Refinement timed out.")
        finally:
            signal.alarm(0)
    elif sys.argv[2] == "comp_methods":
        if len(args) != 3:
            print(
                "Usage: python run.py comp_methods <args refinement> <> <args no refinement without -ss and -sc> <> <args regression>",
                args,
                sys.argv,
            )
            sys.exit(1)
        ref_parser = ref_args_parser()

        ref_args = ref_parser.parse_args(args[0])

        signal.signal(signal.SIGALRM, timeout_handler)
        print("Starting timeout", timeout)
        try:
            signal.alarm(timeout)
            ref_stats = ref_main(ref_args)
        except TimeoutError:
            print("Refinement timed out, stopping experiment.")
            exit(1)
        finally:
            signal.alarm(0)

        samples = ref_stats["sample_count"]

        print("Samples from refinement: ", samples)

        ref_args_2 = ref_parser.parse_args(args[1])
        ref_args_2.stopping_samples = samples
        ref_args_2.stopping_criteria = "samples"
        signal.alarm(timeout)
        try:
            ref_main(ref_args_2)
        except TimeoutError:
            print("No-refinement timed out, continue to regression.")
        finally:
            signal.alarm(0)

        reg_parser = reg_argsparser()
        reg_args = reg_parser.parse_args(args[2])
        reg_args.amount = samples
        signal.alarm(timeout)
        try:
            reg_main(reg_args)
        except TimeoutError:
            print("Regression timed out.")
        finally:
            signal.alarm(0)
