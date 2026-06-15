import os

from signal_matcher.sound import Sound
from signal_matcher.solver import Solver
from signal_matcher.match import Result

class SongSolver(Solver):

    def __init__(self, ref_dir: str) -> None:
        with os.scandir(ref_dir) as files:
            for file in files:
                if file.is_file():
                    self.references.append(Sound(filepath=file.path))

    def sliding_window_solve(self, sample: Sound,
            compare_method: Callable[[Signal, Signal], float], visualize: bool = False) -> Result:
        self.samples = [ sample ]
        results = super().sliding_window_solve(compare_method, visualize)
        return results.data[sample]

    def cross_correlation_solve(self, sample: Sound, compare_method: Callable[[Signal, Signal], float],
            corr_method: str = 'fft', visualize: bool = False) -> Result:
        self.samples = [ sample ]
        results = super().cross_correlation_solve(compare_method, corr_method, visualize)
        return results.data[sample]

