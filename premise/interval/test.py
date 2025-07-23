from premise.interval.maximal_likelihood import *

args.mc = 'airportA-7-10-10'

suo, initial_amount, horizon = build_suo(args)



mle_learning(
    suo: SystemUnderObservation,
    all_states: list[State],
    all_transitions: list[tuple[State, State]],
    all_initial_states: list[State],
    iterations: int,
    initial_length: int,
    horizon: int,
    learning_amount: int,
    model_path: str,
):
