import os
import numpy as np
from functools import total_ordering
import csv

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

    def time_window_to_str(self) -> str:
        return f'{self.time_window[0]} - {self.time_window[-1]}'

    @staticmethod
    def time_window_from_str(s: str) -> np.array:
        start, end = map(float, s.split("-"))
        return np.arange(start, end + 1, 1, dtype=float)


class Result():
    best_match: Match
    second_best_match: Match
    time_to_compute: int # measured in nanoseconds

    def __init__(self, best: Match, second_best: Match, time: int) -> None:
        self.best_match = best
        self.second_best_match = second_best
        self.time_to_compute = time

    def plot(self, sample: Signal, filepath: str = '', title: str = '') -> None:
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
        fig.suptitle(f"{title}Sample {sample.name}")

        if filepath:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            fig.savefig(filepath)
        else:
            plt.show()
        plt.close(fig)

class ResultSet():
    # Collection of samples paired with their top two matches
    data: dict[Signal, Result]

    def __init__(self) -> None:
        self.data = {}

    def to_csv(self, filepath: str) -> None:
        content: list[list[str]] = [[
            'Sample',
            'Match 1 Reference',
            'Match 1 Confidence',
            'Match 1 Window',
            'Match 2 Reference',
            'Match 2 Confidence',
            'Match 2 Window',
            'Time to Compute (ns)'
        ]]
        for sample, result in self.data.items():
            content.append([
                sample.name,
                result.best_match.reference.name,
                str(result.best_match.confidence),
                result.best_match.time_window_to_str(),
                result.second_best_match.reference.name,
                str(result.second_best_match.confidence),
                result.second_best_match.time_window_to_str(),
                str(result.time_to_compute)
            ])

        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerows(content)

    # actual signal data is lost on save, just loads filenames
    def from_csv(self, filepath: str) -> None:
        with open(filepath, newline="") as file:
            reader = csv.reader(file)
            # skip headers
            headers = next(reader)
            for row in reader:
                signal: Signal = Signal()
                signal.name = row[0]

                best_match_ref: Signal = Signal()
                best_match_ref.name = row[1]
                best_match_time_window: np.array = Match.time_window_from_str(row[3])
                best_match: Match = Match(best_match_ref, float(row[2]), best_match_time_window)

                second_best_match_ref: Signal = Signal()
                second_best_match_ref.name = row[4]
                second_best_match_time_window: np.array = Match.time_window_from_str(row[6])
                second_best_match: Match = Match(second_best_match_ref, float(row[5]), second_best_match_time_window)

                time_to_compute: int = int(row[7])

                self.data[signal] = Result(best_match, second_best_match, time_to_compute)

