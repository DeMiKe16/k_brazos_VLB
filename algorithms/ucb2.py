
import numpy as np

from algorithms.algorithm import Algorithm

class UCB2(Algorithm):

    def __init__(self, k: int, alpha: float = 0.5):
        """
        Inicializa el algoritmo UCB2.
        :param k: Número de brazos.
        :param alpha: Parámetro de ajuste para el balance entre exploración y explotación. (0 < alpha < 1).
        :raises ValueError: Si alpha no es positivo o es mayor que 1.
        """
        assert 0 < alpha < 1, "El parámetro alpha debe ser mayor que 0 y menor que 1."

        super().__init__(k)
        self.alpha = alpha
        self.epochs: np.ndarray = np.zeros(k, dtype=int)
        self.selected_arm = None
        self.rounds_left = 0

    def calc_tau(self, i: int):
        return np.ceil((1 + self.alpha)** self.epochs[i])
    
    def calc_nextTau(self, i:int):
        return np.ceil((1 + self.alpha)** (self.epochs[i]+1))

    def select_arm(self) -> int:
        """
        Selecciona un brazo basado en la política UCB2.

        :return: índice del brazo seleccionado.
        """

        if self.selected_arm is not None and self.rounds_left > 0:
            self.rounds_left -=1
            return self.select_arm

        for arm in range(self.k):
            if self.counts[arm] == 0:
                self.selected_arm = arm
                tau = self.calc_tau(arm)
                self.rounds_left = tau - self.epochs[arm]
                return arm
        
        t = np.sum(self.counts)
        for arm in range(self.k):
            tau = self.calc_tau(arm)
            ucb2 = self.values + np.sqrt(((1+self.alpha)*np.log((np.e * t)/tau))/2*tau)
            
        chosen_arm = np.argmax(ucb2)
        self.selected_arm = chosen_arm
        tau = self.calc_tau(arm)
        next_tau = self.calc_nextTau(arm)

        self.rounds_left = next_tau - tau

        return chosen_arm