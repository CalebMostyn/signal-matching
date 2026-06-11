import os
import time
from typing import Callable


from signal_matcher.signal import Signal
from signal_matcher.match import Match, Result, ResultSet 
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

def sliding_window_matching(samples: list[Signal], references: list[Signal],
                            compare_method: Callable[[Signal, Signal], float],
                            visualize: bool = False) -> ResultSet:
    result: ResultSet = ResultSet()
    for sample in samples:
        start_ts: int = time.perf_counter_ns()
        # store the best match for each reference for finding the best overall matches
        best_matches: list[Match] = list()
        # Assumes no gaps in time
        window_size: int = len(sample.time)
        for ref in references:
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

if __name__ == "__main__":
    # Load reference signals
    ref_signals: list[Signal] = []
    with os.scandir('data/reference/') as files:
        for file in files:
            if file.is_file():
                ref_signals.append(Signal(filepath=file.path))
    
    # Load experimental signals
    exp_signals: list[Signal] = []
    with os.scandir('data/experimental/') as files:
        for file in files:
            if file.is_file():
                exp_signals.append(Signal(filepath=file.path))

    sliding_window_matching(exp_signals, ref_signals, Signal.nrmse, True)

