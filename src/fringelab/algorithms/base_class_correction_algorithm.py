from abc import ABC, abstractmethod

class CorrectionAlgorithm(ABC):
    @abstractmethod
    def recover_phase(self, points):
        pass