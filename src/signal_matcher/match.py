import numpy as np
from functools import total_ordering

import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt

from signal_matcher.signal import Signal

@total_ordering
class Match():
    # Signal matched to, confidence, and time location of the match
    reference: Signal
    confidence: float
    time_window: np.array

    def __init__(self, reference: Signal, confidence: float, time_window: np.array) -> None:
        self.reference = reference
        self.confidence = confidence
        self.time_window = time_window

    def __eq__(self, other: Match) -> bool:
        return self.confidence == other.confidence

    def __lt__(self, other: Match) -> bool:
        return self.confidence < other.confidence

class Result():
    best_match: Match
    second_best_match: Match
    time_to_compute: int # measured in nanoseconds

    def __init__(self, best: Match, second_best: Match, time: int) -> None:
        self.best_match = best
        self.second_best_match = second_best
        self.time_to_compute = time

    def plot(self, sample: Signal) -> None:
        fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(10, 5), layout='constrained')
        axs[0].plot(self.best_match.reference.time, self.best_match.reference.intensity)
        axs[0].plot(self.best_match.time_window, sample.intensity)
        axs[0].set_title(f"Best Match - {self.best_match.reference.name}")
        axs[0].text(
            0.98, 0.98, f"Confidence: {self.best_match.confidence:.5f}",
            transform=axs[0].transAxes,
            ha='right', va='top',
            fontsize=10,
            bbox=dict(facecolor='white', alpha=0.6, edgecolor='none')
        )
        axs[0].set_xlabel("Time")
        axs[0].set_ylabel("Intensity")

        axs[1].plot(self.second_best_match.reference.time, self.second_best_match.reference.intensity)
        axs[1].plot(self.second_best_match.time_window, sample.intensity)
        axs[1].set_title(f"Second Best Match - {self.second_best_match.reference.name}")
        axs[1].text(
            0.98, 0.98, f"Confidence: {self.second_best_match.confidence:.5f}",
            transform=axs[1].transAxes,
            ha='right', va='top',
            fontsize=10,
            bbox=dict(facecolor='white', alpha=0.6, edgecolor='none')
        )
        axs[1].set_xlabel("Time")
        axs[1].set_ylabel("Intensity")

        fig.text(
            0.5,
            0.01,
            f"Time to Compute: {(self.time_to_compute / 1_000_000):.5f} ms",
            ha="center",
            va="bottom"
        )

        plt.tight_layout(rect=(0, 0, 1, 0.95))
        fig.suptitle(f"Sample {sample.name}")
        plt.show()

class ResultSet():
    # Collection of samples paired with their top two matches
    data: dict[Signal, Result] = dict()

