
import numpy as np

from algorithms.algorithm import Algorithm

class UCB1(Algorithm):

    def __init__(self, k: int):
        """
        Inicializa el algoritmo UCB1.
        :param k: Número de brazos.
        """
        super().__init__(k)

    def select_arm(self) -> int:
        """
        Selecciona un brazo basado en la política UCB1.
        :return: índice del brazo seleccionado.
        """
        for arm in range(self.k):
            if self.counts[arm] == 0:
                return arm
            
        t = np.sum(self.counts)
        for arm in range(self.k):
            ucb1 = self.values + np.sqrt((2*np.log(t))/self.counts[arm])
            
        chosen_arm = np.argmax(ucb1)

        return chosen_arm