from typing import Callable

from signal_matcher.signal import Signal
from signal_matcher.match import ResultSet
from signal_matcher.solver import Solver

if __name__ == "__main__":

    solver: Solver = Solver('data/experimental', 'data/reference')
    compare_methods: list[Callable[[Signal, Signal], float]] = [
        Signal.euclidian_distance,
        Signal.rmse,
        Signal.nrmse
    ]
    for method in compare_methods:
        results: ResultSet = solver.sliding_window_solve(method)
        results.to_csv(f'results/{method.__name__}.csv')

