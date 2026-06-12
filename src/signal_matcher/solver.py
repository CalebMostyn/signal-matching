import os
import time
from typing import Callable

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy.signal import correlate, correlation_lags

from signal_matcher.signal import Signal
from signal_matcher.match import Match, Result, ResultSet 

class Solver():
    samples: list[Signal] = []
    references: list[Signal] = []

    def __init__(self, sample_dir: str, ref_dir: str) -> None:
        self.load_data(sample_dir, ref_dir)

    def load_data(self, sample_dir: str, ref_dir: str) -> None:
        # Load sample signals
        with os.scandir(sample_dir) as files:
            for file in files:
                if file.is_file():
                    self.samples.append(Signal(filepath=file.path))

        # Load reference signals
        with os.scandir(ref_dir) as files:
            for file in files:
                if file.is_file():
                    self.references.append(Signal(filepath=file.path))

    def sliding_window_solve(self, compare_method: Callable[[Signal, Signal], float],
                                visualize: bool = False) -> ResultSet:
        result: ResultSet = ResultSet()
        for sample in self.samples:
            start_ts: int = time.perf_counter_ns()
            # store the best match for each reference for finding the best overall matches
            best_matches: list[Match] = list()
            # Assumes no gaps in time
            window_size: int = len(sample.time)
            for ref in self.references:
                # Compare points in experimental signal to points in the reference,
                # sliding over all possible time windows as the sample is
                # shorter than the reference
                intensity_windows = sliding_window_view(ref.intensity, window_shape=window_size)
                time_windows = sliding_window_view(ref.time, window_shape=window_size)
    
                best_match: Match = None
                for ii in range(0, len(time_windows)):
                    windowed_ref = Signal(time=time_windows[ii], intensity=intensity_windows[ii])
                    confidence = compare_method(sample, windowed_ref)
                    match: Match = Match(ref, confidence, time_windows[ii])
                    # compare this match's confidence to current best
                    if best_match is None or match >= best_match:
                        best_match = match
    
                # best match for this reference has been found
                best_matches.append(best_match)
    
            # sort in descending order to find confidence closest to 1.0
            best_matches.sort(reverse=True)
            
            # store time to calculate
            time_to_calculate: int = time.perf_counter_ns() - start_ts
    
            # store best matches in result
            result.data[sample] = Result(best_matches[0], best_matches[1], time_to_calculate)
    
            # plot two best matches
            if visualize:
                result.data[sample].plot(sample)
        return result

    def cross_correlation_solve(self, compare_method: Callable[[Signal, Signal], float],
                                 corr_method: str = 'fft', visualize: bool = False) -> ResultSet:
        result: ResultSet = ResultSet()
        for sample in self.samples:
            start_ts: int = time.perf_counter_ns()
            # store the best match for each reference for finding the best overall matches
            best_matches: list[Match] = list()
            for ref in self.references:

                # compute cross correlation on z-score normalized data
                corr: np.array = correlate(
                    (ref.intensity - np.mean(ref.intensity)) / np.std(ref.intensity),
                    (sample.intensity - np.mean(sample.intensity)) / np.std(sample.intensity),
                    mode='valid',
                    method=corr_method
                )

                # find best match
                lags: np.array = correlation_lags(len(ref.intensity), len(sample.intensity), mode='valid')
                # find window in reference from match
                start: int = lags[np.argmax(corr)]
                time_window: np.array = np.arange(start, start + len(sample.time), 1, dtype=float)
                intensity_window: np.array = ref.intensity[np.searchsorted(ref.time, time_window)]
                # compute confidence once on best match
                sub_signal: Signal = Signal(intensity=intensity_window, time=time_window)
                confidence: float = compare_method(sample, sub_signal)

                # store match for this reference signal
                match: Match = Match(ref, confidence, time_window)
                best_matches.append(match)

            # sort in descending order to find confidence closest to 1.0
            best_matches.sort(reverse=True)
            
            # store time to calculate
            time_to_calculate: int = time.perf_counter_ns() - start_ts
        
            # store best matches in result
            result.data[sample] = Result(best_matches[0], best_matches[1], time_to_calculate)

            # plot two best matches
            if visualize:
                result.data[sample].plot(sample)
        return result

