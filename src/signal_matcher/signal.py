import os
import numpy as np

class Signal():
    time: np.array = None
    intensity: np.array = None
    name: string = ""

    def __init__(self, filepath: str = "", time: np.array = None, intensity: np.array = None) -> None:
        if filepath != "":
            self.name = os.path.basename(filepath)
            self.time, self.intensity = np.loadtxt(filepath, delimiter=",", skiprows=1, unpack=True)
        else:
            if not time is None:
                self.time = time
            if not intensity is None:
                self.intensity = intensity

