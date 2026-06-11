import numpy as np

class Signal():
    time: np.array
    intensity: np.array

    def __init__(self, filename: str):
        self.time, self.intensity = np.loadtxt(filename, delimiter=",", skiprows=1, unpack=True)
