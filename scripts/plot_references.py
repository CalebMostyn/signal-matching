import os

import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
from signal_matcher.signal import Signal

if __name__ == "__main__":
    # Load reference signals
    ref_signals: list[Signal] = []
    with os.scandir('data/reference/') as files:
        for file in files:
            if file.is_file():
                ref_signals.append(Signal(filepath=file.path))
    
    # Plot all the reference signals on a single graph
    for signal in ref_signals:
        plt.plot(signal.time, signal.intensity, label=signal.name)
    
    plt.legend()
    plt.xlabel("Time")
    plt.ylabel("Intensity")
    plt.title("Reference Signals")
    plt.show()
