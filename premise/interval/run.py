import sys
from multiprocessing import Pool
from premise.interval.model_free.regression_model import reg_argsparser, reg_main
from premise.interval.refinement import ref_args_parser, ref_main


def split_args(args, delim):
    res = [[]]
    for a in args:
        if a == delim:
            res.append([])
        else:
            res[-1].append(a)
    return res


def run_with_timeout(func, args, timeout):
    with Pool(processes=1) as pool:
        async_result = pool.apply_async(func, args)
        try:
            result = async_result.get(timeout)
        except Exception as e:
            pool.terminate()
            pool.join()
            if isinstance(e, TimeoutError):
                print("Timeout occurred")
                raise TimeoutError("Timeout occurred")
            raise e
        return result


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
        try:
            run_with_timeout(ref_main, (ref_args,), timeout)
        except TimeoutError:
            print("Refinement timed out.")
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
        print("Starting timeout", timeout)
        try:
            ref_stats = run_with_timeout(ref_main, (ref_args,), timeout)
        except TimeoutError:
            print("Refinement timed out, stopping experiment.")
            exit(1)

        transition_count = ref_stats["transition_count"]
        print("Samples from refinement: ", transition_count)

        ref_args_2 = ref_parser.parse_args(args[1])
        ref_args_2.stopping_samples = transition_count
        ref_args_2.stopping_criteria = "samples"
        try:
            run_with_timeout(ref_main, (ref_args_2,), timeout)
        except TimeoutError:
            print("No-refinement timed out, continue to regression.")

        reg_parser = reg_argsparser()
        reg_args = reg_parser.parse_args(args[2])
        reg_args.amount = transition_count // reg_args.length
        try:
            run_with_timeout(reg_main, (reg_args,), timeout)
        except TimeoutError:
            print("Regression timed out.")
