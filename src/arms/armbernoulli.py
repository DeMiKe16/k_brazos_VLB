

import numpy as np

from arms import Arm


class ArmBernoulli(Arm):
    def __init__(self, p: float):
        """
        Inicializa el brazo con una distribución de Bernoulli.

        :param p: Probabilidad de éxito del brazo (valor entre 0 y 1).
        """
        assert p >= 0.0
        assert p <= 1.0
        self.p  = p

    def pull(self):
        """
        Genera una recompensa siguiendo una distribución de bernoulli.

        :return: Recompensa obtenida del brazo.
        """
        return np.random.binomial(1, self.p)
        

    def get_expected_value(self) -> float:
      """
      Devuelve el valor esperado de la distribución Bernoulli.
      
      :return: Probabilidad de éxito.
      """
      return self.p

    def __str__(self):
        """
        Representación en cadena del brazo bernoulli.

        :return: Descripción detallada del brazo bernoulli.
        """
        return f"ArmBenoulli(p={self.p})"

    @classmethod
    def generate_arms(cls, k: int, p_min: float = 0.0, p_max: float = 1.0):
        """
        Genera `k` brazos con probabilidades únicas en el rango [p_min, p_max].

        :param k: Número de brazos a generar.
        :param p_min: Valor mínimo de la probabilidad de éxito.
        :param p_max: Valor máximo de la probabilidad de éxito.
        :return: Lista de brazos generados.
        """
        assert k > 0, "El número de brazos k debe ser mayor que 0."
        assert p_min < p_max, "El valor de mu_min debe ser menor que mu_max."

        # Generar k- valores únicos de mu con decimales
        p_values = set()
        while len(p_values) < k:
            p = np.random.uniform(p_min, p_max)
            p = round(p, 2)
            p_values.add(p)

        p_values = list(p_values)
  

        arms = [ArmBernoulli(p) for p in p_values]

        return arms


