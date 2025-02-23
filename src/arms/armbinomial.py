from numpy.random import binomial


import numpy as np

from arms import Arm
class ArmBinomial(Arm):
    """
    Implementación de un Bandido Multibrazo (Multi-Armed Bandit) basado
    en una distribución binomial

    Parámetros
    ----------
    number: integer
        Número de recompensas que puede devolver el agente
    probability : float
        Probabilidad de que el objeto devuelva una recompensa
    
    Métodos
    -------
    pull :
        Realiza una tirada en el bandido
        
    """
    def __init__(self, probability, number=2):

        """
        Inicializa un objeto con una probabilidad y un número asociado.

        :param probability: Probabilidad de éxito (debe estar en el rango [0, 1]).
        :param number: Un número positivo asociado al objeto (por defecto, 2).
        """
        assert probability >= 0.0
        assert probability <= 1.0
        assert number > 0  
        self.number = number
        self.probability = probability
        
        
    def pull(self):
        """ Realiza una tirada en el bandido

        Retorna
        -------
        reward: float
            Recompensa obtenida en la tirada
        """  
        return binomial(self.number, self.probability)

    def get_expected_value(self) -> float:
        """
        

        :return: Valor esperado de la distribución.
        """

        return self.probability * self.number
    def __str__(self) -> str:
        return f"ArmBinomial(probability={self.probability}, number={self.number})"

    @classmethod
    def generate_arms(cls, k: int, p_min: float = 0.0, p_max: float = 1.0, n: int = 2):
        """
        Genera `k` brazos con probabilidades únicas en el rango [p_min, p_max] y número de ensayos `n` en una distribución binomial.

        :param k: Número de brazos a generar.
        :param p_min: Valor mínimo de la probabilidad de éxito (debe estar en el rango [0, 1]).
        :param p_max: Valor máximo de la probabilidad de éxito (debe estar en el rango [0, 1]).
        :param n: Número de ensayos en la distribución binomial (debe ser mayor que 0).
        :return: Lista de brazos generados.
        """
        assert k > 0, "El número de brazos k debe ser mayor que 0."
        assert n > 0, "El n es demasiado bajo"
        assert p_min < p_max, "El valor de mu_min debe ser menor que mu_max."

        # Generar k- valores únicos de mu con decimales
        p_values = set()
        while len(p_values) < k:
            p = np.random.uniform(p_min, p_max)
            p = round(p, 2)
            p_values.add(p)

        p_values = list(p_values)
       

        arms = [ArmBinomial(p, n) for p in p_values]

        return arms
  