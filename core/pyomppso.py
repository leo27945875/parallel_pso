import time
import timeit
import random
import numpy as np
import ompPSO
from typing import Iterable

from .funcs import *
from .animation import *


class PSO_OMP:
    def __init__(
        self, 
        func     : Callable[[np.ndarray], float],
        dim      : int   = 2,
        n        : int   = 20,
        iters    : int   = 200,
        c0       : float = 2.,
        c1       : float = 2.,
        w_max    : float = 1.,
        w_min    : float = 0.1,
        v_max    : float = float("inf"),
        x_max    : float = 5.,
        x_min    : float = -5.,
        n_thread : int   = 4
    ) -> None:
        
        ompPSO.set_num_threads(n_thread)

        self.func  = func
        self.dim   = dim
        self.n     = n
        self.iters = iters
        self.c0    = c0
        self.c1    = c1
        self.w     = w_max
        self.w_max = w_max
        self.w_min = w_min
        self.v_max = v_max
        self.x_max = x_max
        self.x_min = x_min

        self.xs = np.array([x_min + (x_max - x_min) * np.random.rand(self.dim) for _ in range(n)])
        self.x_fits = np.zeros(n)
        ompPSO.calc_fitness_vals(self.xs, self.x_fits)

        self.local_best_xs = self.xs.copy()
        self.local_best_fits = self.x_fits.copy()

        self.global_best_x, self.global_best_fit = self.init_global_info()
        self.vs = np.zeros((n, self.dim))

    def run(self, verbose: int = 0) -> tuple[np.ndarray, float]:
        if verbose:
            self.print_init_info(verbose)
        for i in range(self.iters):
            self.step(i, verbose)
        if verbose:
            self.print_global_info(verbose)
    
    def step(self, i: int, verbose: int = 1, plt_line: Artist | None = None) -> Iterable[Artist] | None:
        
        ompPSO.update_velocities_and_positions(self.xs, self.vs, self.local_best_xs, self.global_best_x, self.w, self.c0, self.c1, self.x_min, self.x_max, -self.v_max, self.v_max)
        ompPSO.calc_fitness_vals(self.xs, self.x_fits)
        ompPSO.update_best_values(self.xs, self.local_best_xs, self.x_fits, self.local_best_fits, self.global_best_x, self.global_best_fit)

        self.update_inertia_weight(i + 1)
        if verbose:
            self.print_iter_info(i + 1, verbose)
        if plt_line:
            xp, yp, zp = calc_plot_points(self.xs, self.func)
            plt_line.set_data_3d(xp, yp, zp)
            return [plt_line]
    
    def print_init_info(self, verbose: int) -> None:
        print("=" * 100)
        print("Init info:")
        print(f"Basic info : num = {self.n}, dim = {self.dim}, iterations = {self.iters}")
        if (verbose >= 2):
            print(f"Global best point: [{" ".join([f"{float(x):.3e}" for x in self.global_best_x])}]")
        print(f"Global best fitness = {self.global_best_fit}")
        print("=" * 100)
    
    def print_iter_info(self, i: int, verbose: int) -> None:
        print("-" * 50 + f" {i} / {self.iters} " + "-" * 50)
        if (verbose >= 2): 
            print(f"Inertia weight = {self.w}")
            print(f"Global best point: [{" ".join([f"{float(x):.3e}" for x in self.global_best_x])}]")
        print(f"Global best fitness = {self.global_best_fit}")

    def print_global_info(self, verbose: int) -> None:
        print("=" * 100)
        print("Final result:")
        if verbose >= 2:
            print(f"Global best point: [{" ".join([f"{float(x):.3e}" for x in self.global_best_x])}]")
        print(f"Global best fitness = {self.global_best_fit}")
        print("=" * 100)

    def init_global_info(self) -> tuple[np.ndarray, np.ndarray]:
        global_best_idx = np.argmin(self.x_fits)
        return self.xs[global_best_idx], self.x_fits[global_best_idx: global_best_idx + 1]
    
    def update_inertia_weight(self, i: int) -> None:
        self.w = self.w_max - (self.w_max - self.w_min) * (i / self.iters)
    

if __name__ == "__main__":

    func        = levy_func
    seed        = 0
    is_make_ani = True
    n_thread    = 4
    dim         = 2
    n           = 64
    iters       = 50
    x_min       = -10.
    x_max       = 10.
    v_max       = 1.
    markersize  = 4
    n_test      = 10
    verbose     = 2

    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    else:
        random.seed(int(time.time()))
        np.random.seed(int(time.time()))

    pso = PSO_OMP(
        func     = func,
        dim      = dim,
        n        = n,
        iters    = iters,
        x_min    = x_min,
        x_max    = x_max,
        v_max    = v_max,
        n_thread = n_thread
    )

    if is_make_ani:
        assert dim == 2
        fig, ax, surf, line = plot_func(func, pso.xs, x_min, x_max, markersize=markersize, is_show=False)
        make_animation(pso.step, iters, fig, line, verbose, save_path=f"assets/PSO_{func.__name__}--{n=}_{iters=}.gif")
    else:
        t = timeit.timeit(lambda: pso.run(verbose), number=n_test) / n_test
        print(f"Total time = {t}(s)")