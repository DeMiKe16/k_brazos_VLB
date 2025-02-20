
import numpy as np

from algorithms.algorithm import Algorithm

class Softmax(Algorithm):
    def __init__(self, k: int, tau: float = 1.0):
        """
        Inicializa el algoritmo Softmax según el pseudocódigo.
        
        Args:
            k (int): Número de brazos (K en el algoritmo)
            tau (float): Temperatura τ para la función de Gibbs (softmax)
        """
        # Inicialización (línea 1-3 del algoritmo)
        super().__init__(k)
        self.tau = tau  # τ > 0
        self.Q = np.zeros(k)  # Q₀(a) = 0 para cada acción a
    
    def select_arm(self) -> int:
        """
        Selecciona una acción A_t según la distribución de probabilidad π_t(a)
        (línea 8 del algoritmo)
        
        Returns:
            int: Índice del brazo seleccionado
        """
        
        scaled_values = self.Q / self.tau
        
        # Estabilidad numérica
        max_value = np.max(scaled_values)
        exp_values = np.exp(scaled_values - max_value)
        
        return np.random.choice(self.k, p=exp_values / np.sum(exp_values))
    
    def update(self, chosen_arm: int, reward: float):
        """
        Actualiza la recompensa promedio Q_t(a) según la fórmula:
        Q_t(a) = Q_(t-1)(a) + 1/k * (R_t - Q_(t-1)(a))
        (línea 10 del algoritmo)
        
        Args:
            chosen_arm (int): A_t - acción seleccionada
            reward (float): R_t - recompensa recibida
        """
        
        # Actualizar Q_t(a) usando la fórmula del algoritmo
        self.Q[chosen_arm] = self.Q[chosen_arm] + (1.0 / self.k) * (reward - self.Q[chosen_arm])
    
    def reset(self):
        """
        Reinicia el algoritmo a su estado inicial.
        """
        super().reset()
        self.Q = np.zeros(self.k)  # Reiniciar Q₀(a) = 0

