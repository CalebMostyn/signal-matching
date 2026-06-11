import os

import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt

from signal_matcher.signal import Signal
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

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

# euclidian distance
# def calculate_similarity(a: np.array, b: np.array) -> float:
#     distance = np.linalg.norm(a - b)
#     return 1 / (1 + distance)

# root mean square error
# def calculate_similarity(a: np.array, b: np.array) -> float:
#     rmse = np.sqrt(np.mean((a - b) ** 2))
#     return 1 / (1 + rmse)

# normalized root mean square error
def calculate_similarity(a: np.array, b: np.array) -> float:
    rmse = np.sqrt(np.mean((a - b) ** 2))
    scale = np.max(a) - np.min(a)  # range of the signal
    if scale == 0:
        return 1.0 if rmse == 0 else 0.0
    normalized_rmse = rmse / scale
    return 1 / (1 + normalized_rmse)

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

# Match experimental signals to reference signals
for exp in exp_signals:
    similarity_scores: dict[Signal, (float, np.array)] = {}
    # Assumes no gaps in time
    window_size: int = len(exp.time)
    for ref in ref_signals:
        # Compare points in experimental signal to the reference 1 to 1,
        # windowing over all possible time periods as the sample is
        # shorter than the reference
        intensity_windows = sliding_window_view(ref.intensity, window_shape=window_size)
        time_windows = sliding_window_view(ref.time, window_shape=window_size)
        best_similarity: float = -1
        best_signal: Signal = Signal()
        for ii in range(0, len(time_windows)):
            similarity = calculate_similarity(exp.intensity, intensity_windows[ii])
            if similarity > best_similarity:
                best_similarity = similarity
                best_signal = Signal(time=time_windows[ii], intensity=intensity_windows[ii])
                best_signal.name = ref.name
        similarity_scores[ref] = (best_similarity, best_signal.time)
    plot_match(exp, similarity_scores)

