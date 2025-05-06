from premise.interval.interval import State


def get_states_and_transitions() -> (
    tuple[list[State], list[tuple[State, State]], list[State]]
):
    colors = [
        "LLL",
        "LLM",
        "LLH",
        "LML",
        "LMM",
        "LMH",
        "LHL",
        "LHM",
        "LHH",
        "MLL",
        "MLM",
        "MLH",
        "MML",
        "MMM",
        "MMH",
        "MHL",
        "MHM",
        "MHH",
        "HLL",
        "HLM",
        "HLH",
        "HML",
        "HMM",
        "HMH",
        "HHL",
        "HHM",
        "HHH",
    ]

    speed = ["slowest", "slow", "fast", "fastest"]

    distance = ["d10", "d20", "d30", "d40", "d50", "d60", "d70"]

    percived_distance = ["pd10", "pd20", "pd30", "pd40", "pd50", "pd60", "pd70"]

    # collision = [True, False]

    all_states = []
    for x in colors:
        for y in speed:
            for z in distance:
                for v in percived_distance:
                    # for u in collision:
                    all_states.append(((y, z, x, v), (x, v), False))

    all_states = [
        s for s in all_states if not (s[0][1] == "d70" and s[0][0] == "fastest")
    ]
    all_states.append((("collision"), "collision", True))

    all_transitions = []

    for x in all_states[:-1]:
        for y in all_states[:-1]:
            if x[0][2] == y[0][2]:
                if abs(int(x[0][1][1:]) - int(y[0][1][1:])) <= 10 and (
                    int(x[0][1][1:]) >= int(y[0][1][1:])
                ):
                    if (
                        x[0][1] == "d10"
                        or x[0][3] == "pd10"
                        or y[0][1] == "d10"
                        or y[0][3] == "pd10"
                    ):
                        all_transitions.append((x, y))
                    elif x[0][0] == "slowest" and (
                        y[0][0] == "slowest" or y[0][0] == "slow"
                    ):
                        all_transitions.append((x, y))
                    elif x[0][0] == "slow" and y[0][0] != "fastest":
                        all_transitions.append((x, y))
                    elif x[0][0] == "fast" and y[0][0] != "slowest":
                        all_transitions.append((x, y))
                    elif x[0][0] == "fastest" and (
                        y[0][0] == "fast" or y[0][0] == "fastest"
                    ):
                        all_transitions.append((x, y))

    for x in all_states[:-1]:
        all_transitions.append((x, all_states[-1]))

    all_transitions.append((all_states[-1], all_states[-1]))

    initial_states = [s for s in all_states[:-1] if s[0][1] in ["d60", "d70"]]

    return all_states, all_transitions, initial_states


if __name__ == "__main__":
    get_states_and_transitions()
