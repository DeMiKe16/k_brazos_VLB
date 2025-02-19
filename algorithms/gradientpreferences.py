
import numpy as np

from algorithms.algorithm import Algorithm

class GradientPreferences(Algorithm):

    def __init__(self, k: int, alpha: float = 0.1):
        """
        Inicializa el algoritmo de gradiente de preferencias.

        :param k: Número de brazos.
        :param alpha: Tasa de aprendizaje para ajustar las preferencias.
        :raises ValueError: Si alpha no es positivo.
        """
        assert alpha > 0, "El parámetro alpha debe ser mayor que 0."
        
        super().__init__(k)
        self.alpha = alpha
        self.preferences = np.zeros(k)  # Inicializa las preferencias en 0
        self.probabilities = np.ones(k) / k  # Inicializa las probabilidades uniformemente
        self.total_rewards = 0  # Para calcular la recompensa promedio
        self.action_counts = np.zeros(k)  # Contador de selecciones por brazo

    def select_arm(self) -> int:
        """
        Selecciona un brazo basado en las probabilidades (softmax).
        :return: Índice del brazo seleccionado.
        """
        max_preference = np.max(self.preferences)  # Evita overflow numérico
        exp_preferences = np.exp(self.preferences - max_preference)
        self.probabilities = exp_preferences / np.sum(exp_preferences)
        
        chosen_arm = np.random.choice(self.k, p=self.probabilities)
        return chosen_arm
    
    def update(self, chosen_arm: int, reward: float):
        """
        Actualiza las preferencias basadas en la recompensa recibida.

        :param chosen_arm: Índice del brazo seleccionado.
        :param reward: Recompensa obtenida.
        """
        self.total_rewards += reward
        self.action_counts[chosen_arm] += 1
        avg_reward = self.total_rewards / np.sum(self.action_counts)  # Recompensa promedio acumulada

        # Actualiza las preferencias usando el gradiente
        for arm in range(self.k):
            if arm == chosen_arm:
                self.preferences[arm] += self.alpha * (reward - avg_reward) * (1 - self.probabilities[arm])
            else:
                self.preferences[arm] -= self.alpha * (reward - avg_reward) * self.probabilities[arm]




