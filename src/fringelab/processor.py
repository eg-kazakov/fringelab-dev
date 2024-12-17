import numpy as np

from algorithms.base_class_correction_algorithm import CorrectionAlgorithm


class FringeProcessor:
    def __init__(self, correction_algorithm: CorrectionAlgorithm):
        self.correction_algorithm = correction_algorithm

    def recover_phase(self, fringeset):
        phase = self.correction_algorithm.recover_phase(fringeset)
        return phase


    def unwrap_phase(self, phase_df):
        wrapped = np.copy(phase_df)
        vertical_prev = wrapped[0][0]
        vertical_scales = 0
        up_count = 0
        down_count = 0
        for i in range(len(wrapped)):
            horizontal_scales = 0
            wrapped[i][0] += vertical_scales*2*np.pi
            if wrapped[i][0] - vertical_prev > np.pi and down_count == 0:
                down_count += 1
                up_count = 0
                vertical_scales -= 1
                wrapped[i][0] -= 2*np.pi
            elif wrapped[i][0] - vertical_prev < -np.pi and up_count == 0:
                up_count += 1
                down_count = 0
                vertical_scales += 1
                wrapped[i][0] += 2*np.pi
            vertical_prev = wrapped[i][0]
            horizontal_prev = wrapped[i][0]
            for j in range(1, len(wrapped)):
                wrapped[i][j] += 2*np.pi*(horizontal_scales + vertical_scales)
                if wrapped[i][j] - horizontal_prev > np.pi:
                    horizontal_scales -= 1
                    wrapped[i][j] -= 2*np.pi
                elif wrapped[i][j] - horizontal_prev < -np.pi:
                    horizontal_scales += 1
                    wrapped[i][j] += 2*np.pi
                horizontal_prev = wrapped[i][j]
        return wrapped
