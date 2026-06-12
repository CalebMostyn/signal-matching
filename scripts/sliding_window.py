from typing import Callable

from signal_matcher.signal import Signal
from signal_matcher.match import ResultSet
from signal_matcher.solver import Solver

if __name__ == "__main__":

    solver: Solver = Solver('data/experimental', 'data/reference')
    compare_methods: list[Callable[[Signal, Signal], float]] = [
        Signal.euclidian_distance,
        Signal.rmse,
        Signal.nrmse,
        Signal.pearson_correlation
    ]
    # solve with each comparison method
    for method in compare_methods:
        results: ResultSet = solver.sliding_window_solve(method)
        # generate csv from results
        results.to_csv(f'results/{method.__name__}.csv')
        # generate plots from each result
        for sample, result in results.data.items():
            result.plot(sample, f'plots/{method.__name__}/{sample.name}.png')


