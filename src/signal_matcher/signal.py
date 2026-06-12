import os
import numpy as np

class Signal():
    time: np.array = None
    intensity: np.array = None
    name: str = ""

    def __init__(self, filepath: str = "", time: np.array = None, intensity: np.array = None) -> None:
        if filepath != "":
            self.name = os.path.basename(filepath)
            self.time, self.intensity = np.loadtxt(filepath, delimiter=",", skiprows=1, unpack=True)
        else:
            if not time is None:
                self.time = time
            if not intensity is None:
                self.intensity = intensity

    def euclidian_distance(self, other: Signal) -> float:
        distance = np.linalg.norm(self.intensity - other.intensity)
        return 1 / (1 + distance)

    def rmse(self, other: Signal) -> float:
        rmse = np.sqrt(np.mean((self.intensity - other.intensity) ** 2))
        return 1 / (1 + rmse)

    def nrmse(self, other: Signal) -> float:
        rmse = np.sqrt(np.mean((self.intensity - other.intensity) ** 2))
        scale = np.max(self.intensity) - np.min(other.intensity)  # range of the signal
        if scale == 0:
            return 1.0 if rmse == 0 else 0.0
        normalized_rmse = rmse / scale
        return 1 / (1 + normalized_rmse)

    def pearson_correlation(self, other: Signal) -> float:
        r = np.corrcoef(self.intensity, other.intensity)[0, 1]
        return (r + 1) / 2

