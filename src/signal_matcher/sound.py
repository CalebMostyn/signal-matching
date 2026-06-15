import os
from typing import BinaryIO

import librosa
import numpy as np

import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt

from signal_matcher.signal import Signal

class Sound(Signal):
    def __init__(self, filepath: str = "", audio_buffer: BinaryIO = None) -> None:
        if filepath != "" or not audio_buffer is None:
            self.name = os.path.basename(filepath)
            if filepath != "":
                audio_data, self.sampling_rate = librosa.load(filepath, sr=None)
            else:
                audio_data, self.sampling_rate = librosa.load(audio_buffer, sr=None)

            self.intensity = np.array(audio_data)
            self.time = np.arange(len(audio_data)) / self.sampling_rate

    def plot(self):
        plt.plot(self.time, self.intensity, label=self.name)
        plt.xlabel("Time")
        plt.ylabel("Intensity")
        plt.show()
