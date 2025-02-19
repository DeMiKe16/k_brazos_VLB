
import numpy as np

from algorithms.algorithm import Algorithm

class Softmax(Algorithm):

    def __init__(self, k: int, tau: float = 0.1):
        """
        Inicializa el algoritmo Softmax.

        :param k: Número de brazos.
        :param tau: Parámetro de temperatura para controlar la exploración (tau > 0).
        :raises ValueError: Si tau no es positivo.
        """
        assert 0 <= tau, "El parámetro tau debe ser mayor que 0."

        super().__init__(k)
        self.tau = tau

    def select_arm(self) -> int:
        """
        Selecciona un brazo basado en la política Softmax.
        :return: índice del brazo seleccionado.
        """
        
        preferences = self.values / self.tau
        max_preference = np.max(preferences)  # Evitar overflow numérico
        exp_preferences = np.exp(preferences - max_preference)
        probabilities = exp_preferences / np.sum(exp_preferences)
        
        chosen_arm = np.random.choice(self.k, p=probabilities)

        return chosen_arm




