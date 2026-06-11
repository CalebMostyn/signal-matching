import os
from typing import Callable

import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt

from signal_matcher.signal import Signal
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

def sliding_window_matching(samples: list[Signal], references: list[Signal], compare_method: Callable[[Signal, Signal], float]):
    # Match experimental signals to reference signals
    for sample in samples:
        similarity_scores: dict[Signal, (float, np.array)] = {}
        # Assumes no gaps in time
        window_size: int = len(sample.time)
        for ref in references:
            # Compare points in experimental signal to the reference 1 to 1,
            # windowing over all possible time periods as the sample is
            # shorter than the reference
            intensity_windows = sliding_window_view(ref.intensity, window_shape=window_size)
            time_windows = sliding_window_view(ref.time, window_shape=window_size)
            best_similarity: float = -1
            best_signal: Signal = Signal()
            for ii in range(0, len(time_windows)):
                # similarity = calculate_similarity(exp.intensity, intensity_windows[ii])
                windowed_signal = Signal(time=time_windows[ii], intensity=intensity_windows[ii])
                similarity = compare_method(sample, windowed_signal)
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_signal = windowed_signal
                    best_signal.name = ref.name
            similarity_scores[ref] = (best_similarity, best_signal.time)
        plot_match(sample, similarity_scores)

def plot_match(signal, similarity_scores):
    best_ref = None
    best_time = None
    best_sim = -1
    for ref, data in similarity_scores.items():
        sim, time = data
        if sim > best_sim:
            best_ref = ref
            best_time = time
            best_sim = sim
    plt.plot(best_ref.time, best_ref.intensity, label=f"ref:{best_ref.name}")
    plt.plot(best_time, signal.intensity, label=f"exp:{signal.name}")
    plt.legend()
    plt.show()

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

    sliding_window_matching(exp_signals, ref_signals, Signal.nrmse)

