import sys

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


if __name__ == "__main__":
    args = split_args(sys.argv[2:], "::")
    if sys.argv[1] == "refinement":
        ref_parser = ref_args_parser()
        ref_args = ref_parser.parse_args(args[0])
        ref_main(ref_args)
    elif sys.argv[1] == "comp_methods":
        if len(args) != 3:
            print(
                "Usage: python run.py comp_methods <args refinement> <> <args no refinement without -ss and -sc> <> <args regression>",
                args,
                sys.argv,
            )
            sys.exit(1)
        ref_parser = ref_args_parser()

        ref_args = ref_parser.parse_args(args[0])
        ref_stats = ref_main(ref_args)
        samples = ref_stats["sample_count"]

        print("Samples from refinement: ", samples)

        ref_args_2 = ref_parser.parse_args(args[1])
        ref_args_2.stopping_samples = samples
        ref_args_2.stopping_criteria = "samples"
        ref_main(ref_args_2)

        reg_parser = reg_argsparser()
        reg_args = reg_parser.parse_args(args[2])
        reg_args.amount = samples
        reg_main(reg_args)
