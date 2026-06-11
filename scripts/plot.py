import matplotlib
matplotlib.use('TkAgg')

from matplotlib import pyplot as plt
from signal_matcher.signal import Signal

ref1 = Signal('data/reference/ref_signal0.csv')

plt.plot(ref1.time, ref1.intensity)
plt.show()
