import os
import math
import numpy as np
from scipy.signal import resample_poly

class Signal():
    """
    Generic class for storing signal data.

    Attributes:
        intensity : np.array
            1D array containing signal intensity values.
        time : np.array
            1D array containing corresponding time data to intensity values.
        sampling_rate : float
            Sampling rate of signal (i.e. time delta between intensities). Measured in Hz.
        name : str
            Optional name for identifying signal.
    """

    intensity: np.array = None
    time: np.array = None
    sampling_rate : float = 1.0
    name: str = ""

    def __init__(self, filepath: str = "", time: np.array = None, intensity: np.array = None) -> None:
        """
        Initializes a signal, either from a CSV file, directly from time and intensity arrays, or without values at all.

        Args:
            filepath : Path to file to load from. Should contain column 0 with time values and column 1 with intensity values.
            time : 1D array of time values.
            intensity : 1D array of intensity values.
        """
        if filepath != "":
            self.name = os.path.basename(filepath)
            self.time, self.intensity = np.loadtxt(filepath, delimiter=",", skiprows=1, unpack=True)
        else:
            if not time is None:
                self.time = time
            if not intensity is None:
                self.intensity = intensity

    def resample(self, sampling_rate: float) -> None:
        """
        Resamples a signal to a given sampling rate in Hz.

        Args:
            sampling_rate : New sampling rate in Hz.
        """
        # reduce ratio to integers (important for polyphase filtering)
        gcd = math.gcd(int(round(sampling_rate)), int(round(self.sampling_rate)))
        up = int(sampling_rate // gcd)
        down = int(self.sampling_rate // gcd)

        # resample signal
        self.intensity = resample_poly(self.intensity, up, down)
        self.time = np.arange(len(self.intensity)) / sampling_rate

        # update sampling rate
        self.sampling_rate = sampling_rate

    def euclidean_distance(self, other: Signal) -> float:
        """
        Computes similarity between two signals based on distance.
        
        Args:
            other : Signal object to compare to.
        Returns:
            Similarity between 0 and 1.
        """
        distance = np.linalg.norm(self.intensity - other.intensity)
        return 1 / (1 + distance)

    def rmse(self, other: Signal) -> float:
        """
        Computes similarity between two signals based on root mean square error.
        
        Args:
            other : Signal object to compare to.
        Returns:
            Similarity between 0 and 1.
        """
        rmse = np.sqrt(np.mean((self.intensity - other.intensity) ** 2))
        return 1 / (1 + rmse)

    def nrmse(self, other: Signal) -> float:
        """
        Computes similarity between two signals based on normalized root mean square error.
        
        Args:
            other : Signal object to compare to.
        Returns:
            Similarity between 0 and 1.
        """
        rmse = np.sqrt(np.mean((self.intensity - other.intensity) ** 2))
        scale = np.max(self.intensity) - np.min(other.intensity)  # range of the signal
        if scale == 0:
            return 1.0 if rmse == 0 else 0.0
        normalized_rmse = rmse / scale
        return 1 / (1 + normalized_rmse)

    def pearson_correlation(self, other: Signal) -> float:
        """
        Computes similarity between two signals based on the Pearson correlation.
        
        Args:
            other : Signal object to compare to.
        Returns:
            Similarity between 0 and 1.
        """
        r = np.corrcoef(self.intensity, other.intensity)[0, 1]
        return (r + 1) / 2

