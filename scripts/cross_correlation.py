from signal_matcher.signal import Signal
from signal_matcher.match import Match, Result, ResultSet
from signal_matcher.solver import Solver

if __name__ == "__main__":

    solver: Solver = Solver('data/experimental', 'data/reference')

    results: ResultSet = solver.cross_correlation_solver(Signal.nrmse, corr_method='direct')
    # generate csv from results
    results.to_csv(f'results/cross_correlation_sliding.csv')
    # generate plots from each result
    for sample, result in results.data.items():
        result.plot(sample, f'plots/cross_correlation_sliding/{sample.name}.png')

    results: ResultSet = solver.cross_correlation_solver(Signal.nrmse, corr_method='fft')
    # generate csv from results
    results.to_csv(f'results/cross_correlation_fft.csv')
    # generate plots from each result
    for sample, result in results.data.items():
        result.plot(sample, f'plots/cross_correlation_fft/{sample.name}.png')

