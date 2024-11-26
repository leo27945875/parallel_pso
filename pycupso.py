import cuPSO

import timeit
import random
import numpy as np
from typing import Callable, Iterable

from funcs import *
from plot  import *


class PSO_CUDA:
    def __init__(
        self, 
        dim    : int          = 2,
        n      : int          = 20,
        iters  : int          = 200,
        c0     : float        = 2.,
        c1     : float        = 2.,
        w_max  : float        = 1.,
        w_min  : float        = 0.1,
        v_max  : float        = float("inf"),
        x_max  : float        = 5.,
        x_min  : float        = -5.,
        seed   : int          = 0,
        device : cuPSO.Device = cuPSO.Device.GPU
    ) -> None:
        self.func   = cuPSO.calc_fitness_val_npy
        self.dim    = dim
        self.n      = n
        self.iters  = iters
        self.c0     = c0
        self.c1     = c1
        self.w      = w_max
        self.w_max  = w_max
        self.w_min  = w_min
        self.v_max  = v_max
        self.x_max  = x_max
        self.x_min  = x_min
        self.device = device

        # Create host buffers:
        self.xs,              \
        self.vs,              \
        self.local_best_xs,   \
        self.global_best_x,   \
        self.x_fits,          \
        self.local_best_fits, \
        self.global_best_fit, \
        self.v_sum_pow2       = self.init_host_buffers()

        # Create device buffers:
        self.d_xs,              \
        self.d_vs,              \
        self.d_local_best_xs,   \
        self.d_global_best_x,   \
        self.d_x_fits,          \
        self.d_local_best_fits, \
        self.d_global_best_fit, \
        self.d_v_sum_pow2       = self.init_device_buffers()

        # Device RNGs:
        self.rng_states = cuPSO.CURANDStates(self.n * self.dim, seed)
        
    def init_host_buffers(self) -> tuple[np.ndarray, ...]:
        xs              = np.array([self.x_min + (self.x_max - self.x_min) * np.random.rand(self.dim) for _ in range(self.n)])
        x_fits          = np.zeros(self.n)
        vs              = np.zeros((self.n, self.dim))
        v_sum_pow2      = np.zeros(self.n)
        cuPSO.calc_fitness_vals_npy(xs, x_fits)
        local_best_xs   = xs.copy()
        local_best_fits = x_fits.copy()
        global_best_x, \
        global_best_fit = self.init_global_info(xs, x_fits)
        return xs, vs, local_best_xs, global_best_x, x_fits, local_best_fits, global_best_fit, v_sum_pow2

    def init_device_buffers(self) -> tuple[cuPSO.Buffer, ...]:
        xs              = cuPSO.Buffer(self.n, self.dim, self.device)
        x_fits          = cuPSO.Buffer(self.n, 1       , self.device)
        vs              = cuPSO.Buffer(self.n, self.dim, self.device)
        v_sum_pow2      = cuPSO.Buffer(self.n, 1       , self.device)
        local_best_xs   = cuPSO.Buffer(self.n, self.dim, self.device)
        local_best_fits = cuPSO.Buffer(self.n, 1       , self.device)
        global_best_x   = cuPSO.Buffer(1     , self.dim, self.device)
        global_best_fit = cuPSO.Buffer(1     , 1       , self.device)
        xs             .copy_from_numpy(self.xs             )
        vs             .copy_from_numpy(self.vs             )
        x_fits         .copy_from_numpy(self.x_fits         )
        local_best_xs  .copy_from_numpy(self.local_best_xs  )
        local_best_fits.copy_from_numpy(self.local_best_fits)
        global_best_x  .copy_from_numpy(self.global_best_x  )
        global_best_fit.copy_from_numpy(self.global_best_fit)
        return xs, vs, local_best_xs, global_best_x, x_fits, local_best_fits, global_best_fit, v_sum_pow2

    def init_global_info(self, xs: np.ndarray, x_fits: np.ndarray) -> tuple[np.ndarray, float]:
        global_best_idx = np.argmin(x_fits)
        return xs[global_best_idx], x_fits[global_best_idx]

    def run(self, verbose: int = 1) -> tuple[np.ndarray, float]:
        for i in range(self.iters):
            self.step(i, verbose)
        self.d_global_best_x.copy_to_numpy(self.global_best_x)
        self.d_global_best_fit.copy_to_numpy(self.global_best_fit)
        return self.global_best_x, self.global_best_fit
    
    def step(self, i: int, verbose: int = 1, plt_line: Artist | None = None) -> Iterable[Artist] | None:
        self.update_velocities()
        self.update_positions()
        self.update_bests()
        self.update_inertia_weight(i + 1)
        if verbose:
            self.print_iter_info(i + 1)
        if plt_line:
            xp, yp, zp = calc_plot_points(self.xs, self.func)
            plt_line.set_data_3d(xp, yp, zp)
            return [plt_line]
    
    def print_iter_info(self, i: int) -> None:
        self.d_global_best_x.copy_to_numpy(self.global_best_x)
        self.d_global_best_fit.copy_to_numpy(self.global_best_fit)
        print("-" * 50 + f" {i} " + "-" * 50)
        print(f"Inertia weight = {self.w}")
        print(f"Global best point: {[round(float(x), 4) for x in self.global_best_x]}")
        print(f"Global best fitness = {self.global_best_fit}")

    def print_global_info(self) -> None:
        self.d_global_best_x.copy_to_numpy(self.global_best_x)
        self.d_global_best_fit.copy_to_numpy(self.global_best_fit)
        print(f"Global best point: {[round(float(x), 4) for x in self.global_best_x]}")
        print(f"Global best fitness = {self.global_best_fit}")
    
    def update_velocities(self) -> None:
        cuPSO.update_velocities(
            self.d_xs, self.d_vs, self.d_local_best_xs, self.d_global_best_x, self.d_v_sum_pow2, self.w, self.c0, self.c1, self.v_max, self.rng_states
        )

    def update_positions(self) -> None:
        cuPSO.update_positions(
            self.d_xs, self.d_vs, self.x_min, self.x_max
        )
    
    def update_bests(self) -> None:
        cuPSO.calc_fitness_vals(self.d_xs, self.d_x_fits)
        cuPSO.update_bests(
            self.d_xs, self.d_x_fits, self.d_local_best_xs, self.d_local_best_fits, self.d_global_best_x, self.d_global_best_fit
        )
    
    def update_inertia_weight(self, i: int) -> None:
        self.w = self.w_max - (self.w_max - self.w_min) * (i / self.iters)


def main():
    seed        = None
    dim         = 10
    n           = 500
    iters       = 1000
    x_min       = -20
    x_max       = 20.
    v_max       = 1.
    is_make_ani = False
    markersize  = 4
    verbose     = 1

    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    pso = PSO_CUDA(
        dim   = dim,
        n     = n,
        iters = iters,
        x_min = x_min,
        x_max = x_max,
        v_max = v_max
    )

    if is_make_ani:
        assert dim == 2
        fig, ax, surf, line = plot_func(pso.func, pso.xs, x_min, x_max, markersize=markersize, is_show=False)
        make_animation(pso.step, iters, fig, line, verbose, save_path=f"PSO_{pso.func.__name__}.png")
    else:
        t = timeit.timeit(lambda: pso.run(verbose), number=1)
        print("-" * 100)
        print(f"Total time = {t}")
        if dim == 2: 
            plot_func(pso.func, pso.xs, x_min, x_max, markersize=markersize, is_show=True)
    
    pso.print_global_info()


if __name__ == "__main__":

    main()