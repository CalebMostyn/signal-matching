import os
import numpy as np

class Signal():
    time: np.array
    intensity: np.array
    name: string

    def __init__(self, filepath: str) -> None:
        self.name = os.path.basename(filepath)
        self.time, self.intensity = np.loadtxt(filepath, delimiter=",", skiprows=1, unpack=True)
