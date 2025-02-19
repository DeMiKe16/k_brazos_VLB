"""
Module: plotting/plotting.py
Description: Contiene funciones para generar gráficas de comparación de algoritmos.

Author: Luis Daniel Hernández Molinero
Email: ldaniel@um.es
Date: 2025/01/29

This software is licensed under the GNU General Public License v3.0 (GPL-3.0),
with the additional restriction that it may not be used for commercial purposes.

For more details about GPL-3.0: https://www.gnu.org/licenses/gpl-3.0.html
"""

from typing import List

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from algorithms import Algorithm, EpsilonGreedy


def get_algorithm_label(algo: Algorithm) -> str:
    """
    Genera una etiqueta descriptiva para el algoritmo incluyendo sus parámetros.

    :param algo: Instancia de un algoritmo.
    :type algo: Algorithm
    :return: Cadena descriptiva para el algoritmo.
    :rtype: str
    """
    label = type(algo).__name__
    if isinstance(algo, EpsilonGreedy):
        label += f" (epsilon={algo.epsilon})"
    # elif isinstance(algo, OtroAlgoritmo):
    #     label += f" (parametro={algo.parametro})"
    # Añadir más condiciones para otros algoritmos aquí
    else:
        raise ValueError("El algoritmo debe ser de la clase Algorithm o una subclase.")
    return label


def plot_average_rewards(steps: int, rewards: np.ndarray, algorithms: List[Algorithm]):
    """
    Genera la gráfica de Recompensa Promedio vs Pasos de Tiempo.

    :param steps: Número de pasos de tiempo.
    :param rewards: Matriz de recompensas promedio.
    :param algorithms: Lista de instancias de algoritmos comparados.
    """
    sns.set_theme(style="whitegrid", palette="muted", font_scale=1.2)

    plt.figure(figsize=(14, 7))
    for idx, algo in enumerate(algorithms):
        label = get_algorithm_label(algo)
        plt.plot(range(steps), rewards[idx], label=label, linewidth=2)

    plt.xlabel('Pasos de Tiempo', fontsize=14)
    plt.ylabel('Recompensa Promedio', fontsize=14)
    plt.title('Recompensa Promedio vs Pasos de Tiempo', fontsize=16)
    plt.legend(title='Algoritmos')
    plt.tight_layout()
    plt.show()


def plot_optimal_selections(steps: int, optimal_selections: np.ndarray, algorithms: List[Algorithm]):
    """
    Genera la gráfica de Porcentaje de Selección del Brazo Óptimo vs Pasos de Tiempo.

    :param steps: Número de pasos de tiempo.
    :param optimal_selections: Matriz de porcentaje de selecciones óptimas.
    :param algorithms: Lista de instancias de algoritmos comparados.
    """
    sns.set_theme(style="whitegrid", palette="muted", font_scale=1.2)

    plt.figure(figsize=(14, 7))
    for idx, algo in enumerate(algorithms):
        label = get_algorithm_label(algo)
        plt.plot(range(steps), optimal_selections[idx], label=label, linewidth=2)

    plt.xlabel('Pasos de Tiempo', fontsize=14)
    plt.ylabel('Selección del Brazo Óptimo', fontsize=14)
    plt.title('Selección del Brazo Óptimo vs Pasos de Tiempo', fontsize=16)
    plt.legend(title='Algoritmos')
    plt.tight_layout()
    plt.show()



def plot_arm_statistics(arm_stats, algorithms):
    """
    Genera gráficos separados mostrando la selección de brazos y sus recompensas promedio.

    :param arm_stats: Lista de listas de diccionarios con estadísticas de cada brazo por algoritmo.
    :param algorithms: Lista de instancias de algoritmos comparados.
    """
    num_algorithms = len(algorithms)
    
    # Crear gráficos separados, uno por cada algoritmo
    for algo_index, (stats, algorithm) in enumerate(zip(arm_stats, algorithms)):
        fig, ax = plt.subplots(figsize=(8, 6))  # Crear una nueva figura para cada algoritmo
        
        arms = [arm["arm"] for arm in stats]
        mean_rewards = [arm["promedy_rewards"] for arm in stats]
        times_pulled = [arm["times_pulled"] for arm in stats]
        optimal_flags = [arm["optimal"] for arm in stats]

        # Etiquetas del eje X: "Nombre del Brazo - Veces seleccionado - (Óptimo o No)"
        x_labels = [f"Brazo {arm} - {round((times/1000)*100, 2)} %- {'Óptimo' if opt else 'No'}" 
                    for arm, times, opt in zip(arms, times_pulled, optimal_flags)]

        # Colores: Verde si el brazo es óptimo, azul si no lo es
        colors = ["green" if opt else "blue" for opt in optimal_flags]

        # Crear el histograma de barras
        ax.bar(x_labels, mean_rewards, color=colors)
       

        # Configuración del gráfico
        ax.set_xlabel("Selecciones del Brazo")
        ax.set_ylabel("Promedio de Ganancias")
        ax.set_title(f"Estadísticas de brazos - {algorithm.__class__.__name__}")
        ax.set_xticklabels(x_labels, rotation=45, ha="right", fontsize=10)

        # Mostrar el gráfico
        plt.tight_layout()
        plt.show()
        for i in range(1, 11):
            print("Recompensa Promedio del Brazo", i, ":", mean_rewards[i-1])

def plot_regret(steps: int, regret_accumulated: np.ndarray,algorithms: List[Algorithm]):
  """
  Genera la gráfica de Regret Acumulado vs Pasos de Tiempo
  :param steps: Número de pasos de tiempo.
  :param regret_accumulated: Matriz de regret acumulado (algoritmos x pasos).
  :param algorithms: Lista de instancias de algoritmos comparados.
 
  """
  sns.set_theme(style="whitegrid", palette="muted", font_scale=1.2)

  plt.figure(figsize=(14, 7))
  for idx, algo in enumerate(algorithms):
      label = get_algorithm_label(algo)
      plt.plot(range(steps), regret_accumulated[idx], label=label, linewidth=2)

  plt.xlabel('Pasos de Tiempo', fontsize=14)
  plt.ylabel('Arrepentimiento Promedio', fontsize=14)
  plt.title('Arrepentimento Promedio vs Pasos de Tiempo', fontsize=16)
  plt.legend(title='Algoritmos')
  plt.tight_layout()
  plt.show()

