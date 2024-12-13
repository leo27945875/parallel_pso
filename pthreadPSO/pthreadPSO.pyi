from __future__ import annotations
import numpy

__all__ = ['set_num_threads', 'calc_fitness_vals', 'update_best_values', 'update_velocities_and_positions']

def set_num_threads(n: int) -> None:
    ...
def calc_fitness_vals(array: numpy.ndarray[numpy.float64], result_array: numpy.ndarray[numpy.float64]) -> None:
    ...
def update_best_values(positions: numpy.ndarray[numpy.float64], p_best_positions: numpy.ndarray[numpy.float64], fitness_values: numpy.ndarray[numpy.float64], p_best_values: numpy.ndarray[numpy.float64], g_best_position: numpy.ndarray[numpy.float64], g_best_value: numpy.ndarray[numpy.float64]) -> None:
    ...
def update_velocities_and_positions(positions: numpy.ndarray[numpy.float64], velocities: numpy.ndarray[numpy.float64], p_best_positions: numpy.ndarray[numpy.float64], g_best_position: numpy.ndarray[numpy.float64], W: float, C0: float, C1: float, x_min: float, x_max: float, v_min: float, v_max: float) -> None:
    ...
