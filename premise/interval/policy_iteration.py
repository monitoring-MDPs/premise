from stormpy import (
    SparseIntervalDtmc,
    SparseRationalIntervalDtmc,
    SparseDtmc,
    SparseExactDtmc,
    SparseMatrixBuilder,
    ExactSparseMatrixBuilder,
    StateLabeling,
    SparseExactModelComponents,
    SparseModelComponents,
    Environment,
    Property,
    model_checking,
    ExplicitQuantitativeCheckResult,
    ExplicitExactQuantitativeCheckResult,
    Rational,
)
from stormpy.pycarl.gmp.formula.formula import Formula

from premise.interval.utils import const


def apply_imc_policy(
    model: SparseIntervalDtmc | SparseRationalIntervalDtmc,
    policy: list[tuple[int, ...]],
) -> SparseDtmc | SparseExactDtmc:
    if len(policy) != len(model.states):
        raise ValueError(
            f"Number of states in policy ({len(policy)}) does not match number of states ({len(model.states)})"
        )

    if model.is_exact:
        builder = ExactSparseMatrixBuilder(0, 0, 0, False, False)
    else:
        builder = SparseMatrixBuilder(0, 0, 0, False, False)

    for state, state_policy in zip(model.states, policy):
        new_state_trans = {}
        budget = const(model.is_exact, 1.0)
        state_trans = state.actions[0].transitions
        state_trans_upper_dict = {}
        for transition in state_trans:
            new_state_trans[transition.column] = transition.value().lower()
            state_trans_upper_dict[transition.column] = transition.value().upper()
            budget -= transition.value().lower()

        if budget < 0:
            raise ValueError(
                f"Budget is negative ({budget}) for statet {state.id} [{new_state_trans}]"
            )

        # The first policy does not contain any transitions
        if state_policy is None:
            state_policy = tuple(new_state_trans.keys())

        for pol_t in state_policy:
            if budget <= 0:
                continue
            elif budget < state_trans_upper_dict[pol_t]:
                new_state_trans[pol_t] += budget
                budget = const(model.is_exact, 0.0)
            else:
                new_state_trans[pol_t] += state_trans_upper_dict[pol_t]
                budget -= state_trans_upper_dict[pol_t]

        for dest, prob in sorted(new_state_trans.items()):
            builder.add_next_value(state.id, dest, prob)

    matrix = builder.build()

    # Build the labeling
    labeling = StateLabeling(len(model.states))
    for label in model.labeling.get_labels():
        labeling.add_label(label)

    for i in range(len(model.states)):
        for label in model.states[i].labels:
            labeling.add_label_to_state(label, i)

    if model.is_exact:
        components = SparseExactModelComponents(matrix, labeling)
        return SparseExactDtmc(components)
    else:
        components = SparseModelComponents(matrix, labeling)
        return SparseDtmc(components)


def policy_from_result(
    model: SparseIntervalDtmc | SparseRationalIntervalDtmc,
    result: list[float] | list[Rational],
    minimize: bool,
):
    new_policy = []
    for state in model.states:
        state_pol = []
        for trans in state.actions[0].transitions:
            state_pol.append((trans.column, result[trans.column]))

        new_policy.append(
            tuple(x[0] for x in sorted(state_pol, key=lambda x: x[1], reverse=minimize))
        )

    return new_policy


def policy_iter_imc(
    imc: SparseIntervalDtmc | SparseRationalIntervalDtmc,
    property: Property,
    extract_policy=False,
    storm_env=Environment(),
):
    policy = [None] * len(imc.states)
    old_policy = None
    result = None
    minimize = "min" in str(property)

    while policy != old_policy:
        mc = apply_imc_policy(imc, policy)
        result = model_checking(mc, property, environment=storm_env)
        old_policy = policy
        policy = policy_from_result(imc, result.get_values(), minimize)

    if extract_policy:
        return result, policy
    return result
