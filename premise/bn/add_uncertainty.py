from argparse import ArgumentParser
from ast import arg
from copy import copy
from itertools import permutations
import math

import stormpy
import stormpy._core
from tqdm import tqdm


def to_vt(value, use_exact: bool):
    if use_exact:
        return stormpy.Rational(value)
    else:
        return float(value)


def load_drn_dtmc(file_path: str, exact_arithmetic: bool):
    opts = stormpy.DirectEncodingParserOptions()
    opts.build_choice_labels = True
    if exact_arithmetic:
        model = stormpy._core._build_sparse_exact_dtmc_from_drn(file_path, opts)
        model = stormpy._convert_sparse_model(model, False)
    else:
        model = stormpy.build_model_from_drn(file_path, opts)
    return model


def load_jani_dtmc(file_path: str, exact_arithmetic: bool):
    jani_program, _ = stormpy.parse_jani_model(file_path)
    expr_manager = jani_program.expression_manager
    options = stormpy.BuilderOptions()
    options.set_build_state_valuations()
    options.set_build_all_labels()
    if exact_arithmetic:
        return (
            stormpy.build_sparse_exact_model_with_options(jani_program, options),
            expr_manager,
        )
    else:
        return (
            stormpy.build_sparse_model_with_options(jani_program, options),
            expr_manager,
        )


def load_prism_dtmc(
    file_path: str, exact_arithmetic: bool, constants: str | None = None
):
    prism_program = stormpy.parse_prism_program(file_path)
    if constants is not None:
        prism_program, _ = stormpy.preprocess_symbolic_input(
            prism_program, [], constants
        )

    options = stormpy.BuilderOptions()
    options.set_build_state_valuations()
    options.set_build_all_labels()
    if exact_arithmetic:
        return stormpy.build_sparse_exact_model_with_options(prism_program, options)
    else:
        return stormpy.build_sparse_model_with_options(prism_program, options)


def load_dtmc(file_path: str, exact_arithmetic: bool, constants: str | None = None):
    if file_path.endswith(".drn"):
        return load_drn_dtmc(file_path, exact_arithmetic)
    elif file_path.endswith(".jani"):
        return load_jani_dtmc(file_path, exact_arithmetic)
    elif file_path.endswith(".pm"):
        return load_prism_dtmc(file_path, exact_arithmetic, constants), None
    else:
        raise ValueError("Unsupported file format. Please use .drn or .jani files.")


def add_uncertainty_to_dtmc(dtmc, uncertainty: float | stormpy.Rational):
    if dtmc.is_exact:
        add_uncertainty = stormpy.AddUncertaintyExact(dtmc)
        min_prob = stormpy.Rational(0.0000001)
    else:
        add_uncertainty = stormpy.AddUncertaintyDouble(dtmc)
        min_prob = 0.0000001

    idtmc = add_uncertainty.transform(uncertainty, min_prob)
    return idtmc


def transform_idtmc_to_mdp(
    idtmc,
    exact: bool,
    expr_manager: stormpy.ExpressionManager | None,
    copy_labels: bool,
):
    if exact:
        builder = stormpy.ExactSparseMatrixBuilder(0, 0, 0, False, True)
    else:
        builder = stormpy.SparseMatrixBuilder(0, 0, 0, False, True)

    variables = []
    variable_values = {}
    if not copy_labels and expr_manager is not None:
        for var in expr_manager.get_variables():
            if var.name[0] != "_" and var.has_integer_type():
                variables.append(var)
                variable_values[var.name] = set()

    state_labels = []
    current_row = 0
    for state in tqdm(idtmc.states):
        if not copy_labels:
            # Set state labels from valuations
            labels = []
            for var in variables:
                val = idtmc.state_valuations._get_integer_value(state, var)
                if val != -1:
                    variable_values[var.name].add(val)
                    labels.append(f"{var.name}{val}")
            state_labels.append(labels)

        # Collect all transitions for the state
        transitions = {}
        for action in state.actions:
            for transition in action.transitions:
                transitions[transition.column] = transition.value()

        # Calculate the minimum probability to ensure every interval has at least a lowerbound
        min_prob = to_vt(0.0, exact)
        for state in transitions:
            min_prob += to_vt(transitions[state].lower(), exact)

        # Build all possible actions based on the transition interval ordernings
        builder.new_row_group(current_row)
        actions = set()

        stack = [(frozenset(), to_vt(1.0, exact) - min_prob, set(transitions.keys()))]
        while stack:
            trans_prob, buget, remaining = stack.pop()
            for state in remaining:
                up = to_vt(transitions[state].upper(), exact)
                low = to_vt(transitions[state].lower(), exact)
                if up <= buget:
                    new_trans_prob = trans_prob.union({(state, up)})
                    new_buget = buget - (up - low)
                    new_remaining = remaining.difference({state})
                    stack.append((new_trans_prob, new_buget, new_remaining))
                else:
                    action_trans_probs = trans_prob.union(
                        [(state, low + buget)]
                        + [
                            (s, to_vt(transitions[s].lower(), exact))
                            for s in remaining
                            if s != state
                        ]
                    )
                    actions.add(frozenset(action_trans_probs))

        # Add all actions to the builder
        for action_trans_probs in actions:
            total = to_vt(0.0, exact)
            for state, prob in sorted(action_trans_probs):
                builder.add_next_value(current_row, state, prob)
                total += prob
            if (
                exact
                and total != 1
                or not exact
                and not math.isclose(total, 1.0, rel_tol=1e-5)
            ):
                raise ValueError(
                    f"Transition probabilities for state {state} do not sum to 1 (sum={total}, diff={total - 1}) [{action_trans_probs}, {transitions}]."
                )
            current_row += 1

    labeling = idtmc.labeling
    state_valuations = idtmc.state_valuations
    new_labeling = stormpy.StateLabeling(idtmc.nr_states)
    del idtmc

    transition_matrix = builder.build()

    # Build state labeling
    if copy_labels:
        new_labeling = labeling
    else:
        new_labeling.add_label("init")

        # Get new labels
        for var_name, values in variable_values.items():
            for val in values:
                label = f"{var_name}{val}"
                new_labeling.add_label(label)

        new_labeling.set_states("init", labeling.get_states("init"))
        for i, labels in enumerate(state_labels):
            for label in labels:
                new_labeling.add_label_to_state(label, i)

    # Build MDP
    if exact:
        components = stormpy.SparseExactModelComponents(transition_matrix, new_labeling)
        components.state_valuations = state_valuations
        return stormpy.SparseExactMdp(components)
    else:
        components = stormpy.SparseModelComponents(transition_matrix, new_labeling)
        components.state_valuations = state_valuations
        return stormpy.SparseMdp(components)


if __name__ == "__main__":
    parser = ArgumentParser(
        description="Add uncertainty to a given DTMC and export either an MDP or iDTMC."
    )
    parser.add_argument("input", type=str, help="Input DTMC file.")
    parser.add_argument("output", type=str, help="Output file (in .drn format).")
    parser.add_argument(
        "--type",
        type=str,
        choices=["mdp", "idtmc"],
        required=True,
        help="Type of model to export: 'mdp' for Markov Decision Process, 'idtmc' for interval DTMC.",
    )
    parser.add_argument(
        "--uncertainty",
        type=float,
        required=True,
        help="Uncertainty value to add to transition probabilities (between 0 and 1).",
    )
    parser.add_argument(
        "--exact", action="store_true", help="Use exact arithmetic for computations."
    )
    parser.add_argument(
        "--float",
        action="store_true",
        help="Use floating-point arithmetic for computations.",
    )
    parser.add_argument(
        "--copy-labels",
        action="store_true",
        help="Copy state labels from the original DTMC to the MDP (default, base labels on state valuations).",
    )
    parser.add_argument(
        "--constants", type=str, help="Constants to apply to prism model."
    )

    args = parser.parse_args()

    if not args.exact and not args.float:
        parser.error("Either --exact or --float must be specified.")

    use_exact = args.exact or not args.float

    dtmc, expr_manager = load_dtmc(args.input, use_exact, args.constants)
    print("Loaded DTMC with", dtmc.nr_states, "states.")
    idtmc = add_uncertainty_to_dtmc(dtmc, args.uncertainty)
    print("Generated iDTMC with", idtmc.nr_states, "states.")

    if args.type == "idtmc":
        stormpy.export_to_drn(idtmc, args.output)
    else:
        mdp = transform_idtmc_to_mdp(idtmc, use_exact, expr_manager, args.copy_labels)
        print("Generated MDP with", mdp.nr_states, "states.")
        stormpy.export_to_drn(mdp, args.output)
