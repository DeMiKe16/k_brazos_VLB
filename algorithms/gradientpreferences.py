
import numpy as np

from algorithms.algorithm import Algorithm

class GradientPreference(Algorithm):
    def __init__(self, k: int, alpha: float):
        """
        Implementación del método de Gradiente de Preferencias según Sutton y Barto (2018).
        
        Args:
            k (int): Número de acciones/brazos
            alpha (float): Tasa de aprendizaje
        """
        super().__init__(k)
        self.alpha = alpha
        self.H = np.zeros(k, dtype=float)  # H_t(a) - preferencias numéricas
        self.average_reward = 0.0  # R̄_t - recompensa promedio
        self.t = 0  # contador de tiempo
        
    def _compute_action_probabilities(self) -> np.ndarray:
        """
        Calcula π_t(a) = e^(H_t(a)) / Σ e^(H_t(b)) para todas las acciones.
        
        Returns:
            np.ndarray: Vector de probabilidades π_t(a)
        """
        # Estabilidad numérica: restamos el máximo antes de exp
        H_stable = self.H - np.max(self.H)
        exp_H = np.exp(H_stable)
        return exp_H / np.sum(exp_H)
        
    def select_arm(self) -> int:
        """
        Selecciona una acción según las probabilidades π_t(a).
        
        Returns:
            int: Índice de la acción seleccionada
        """
        probabilities = self._compute_action_probabilities()
        return np.random.choice(self.k, p=probabilities)
        
    def update(self, chosen_arm: int, reward: float):
        """
        Actualiza las preferencias H_t(a) según las ecuaciones de Sutton y Barto:
        H_(t+1)(A_t) = H_t(A_t) + α(R_t - R̄_t)(1 - π_t(A_t))
        H_(t+1)(a) = H_t(a) - α(R_t - R̄_t)π_t(a) para a ≠ A_t
        
        Args:
            chosen_arm (int): A_t - acción seleccionada
            reward (float): R_t - recompensa recibida
        """
        self.t += 1
        
        # Actualizar promedio de recompensa (R̄_t)
        self.average_reward += (reward - self.average_reward) / self.t
        
        # Calcular probabilidades actuales π_t(a)
        probabilities = self._compute_action_probabilities()
        
        # Calcular el término (R_t - R̄_t)
        reward_diff = reward - self.average_reward
        
        # Actualizar preferencias para todas las acciones
        for arm in range(self.k):
            if arm == chosen_arm:
                # H_(t+1)(A_t) = H_t(A_t) + α(R_t - R̄_t)(1 - π_t(A_t))
                self.H[arm] += self.alpha * reward_diff * (1 - probabilities[arm])
            else:
                # H_(t+1)(a) = H_t(a) - α(R_t - R̄_t)π_t(a)
                self.H[arm] -= self.alpha * reward_diff * probabilities[arm]
    
    def reset(self):
        """
        Reinicia el algoritmo a su estado inicial.
        """
        super().reset()
        self.H = np.zeros(self.k, dtype=float)
        self.average_reward = 0.0
        self.t = 0