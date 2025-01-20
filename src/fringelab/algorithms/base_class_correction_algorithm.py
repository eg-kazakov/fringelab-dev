from abc import ABC, abstractmethod

class CorrectionAlgorithm(ABC):
    @abstractmethod
    def recover_phase(self, I1, I2, I3):
        pass