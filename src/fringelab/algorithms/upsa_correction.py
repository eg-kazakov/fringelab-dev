import numpy as np
from .base_class_correction_algorithm import CorrectionAlgorithm

class UpsaCorrection(CorrectionAlgorithm):

    # Существующий метод получения фазы интерферограмм на основе растяжения эллипса
    # В целом хорошо работает, но точность снижается при наличии шума
    # Принимает три двумерных массива numpy
    # Возвращает одномерный массив с комплексными числами (круговая траектория)
    # Можно привести к двумерному массиву (взять существенную и мнимую часть)
    # Далее можно привести к трехмерному пространству, добавив координату z с нулями

    def recover_phase(self, I1, I2, I3):
        Z1 = np.exp(1j*(2*np.pi/3)*0)*I1 + \
             np.exp(1j*(2*np.pi/3)*1)*I2 + np.exp(1j*(2*np.pi/3)*2)*I3
        Z2 = Z1.real / np.max(abs(Z1.real)) + 1j*(Z1.imag / np.max(abs(Z1.imag)))
        Z3 = np.exp(1j*np.pi/4)*Z2
        Z4 = Z3.real / np.max(abs(Z3.real)) + 1j*(Z3.imag / np.max(abs(Z3.imag)))
        return self._get_phase(Z4)

    # Восстанавливает фазу для метода upsa
    # Принимает одномерный массив комплексных чисел
    # Возвращает двумерный массив с восстановленной фазой
    @staticmethod
    def _get_phase(Z4):
        fi_recovered = np.arctan2(Z4.real, Z4.imag)
        return fi_recovered