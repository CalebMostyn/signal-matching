from signal_matcher.signal import Signal
from signal_matcher.match import Match, Result, ResultSet
from signal_matcher.solver import Solver

if __name__ == "__main__":

    solver: Solver = Solver('data/experimental', 'data/reference')

    # use both sliding time-domain method and frequency domain method
    corr_methods: list[str] = ['direct', 'fft']
    for corr_method in corr_methods:
        results: ResultSet = solver.cross_correlation_solve(Signal.nrmse, corr_method=corr_method)
        # generate csv from results
        results.to_csv(f'results/cross_correlation_{corr_method}.csv')
        # generate plots from each result
        for sample, result in results.data.items():
            result.plot(sample, f'plots/cross_correlation_{corr_method}/{sample.name}.png')

