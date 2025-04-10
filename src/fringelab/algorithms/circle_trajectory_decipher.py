import numpy as np
from .base_class_correction_algorithm import CorrectionAlgorithm

class UpsaCorrection(CorrectionAlgorithm):

    def recover_phase(self, I1, I2, I3):
        fi_recovered = np.arctan2(np.sqrt(3) * (I2 - I3), (I2 + I3 - 2*I1))
        return fi_recovered