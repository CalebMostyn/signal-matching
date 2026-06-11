import numpy as np
from functools import total_ordering

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

class MatchSet():
    # Collection of samples paired with their top two matches
    data: dict[Signal, (Match, Match)] = dict()

