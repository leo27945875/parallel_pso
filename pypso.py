import timeit
import random
import numpy as np
from typing import Callable, Iterable

from funcs import *
from plot  import *


class PSO:
    def __init__(
        self, 
        func  : Callable[[np.ndarray], float],
        dim   : int                           = 2,
        n     : int                           = 20,
        iters : int                           = 200,
        c0    : float                         = 2.,
        c1    : float                         = 2.,
        w_max : float                         = 1.,
        w_min : float                         = 0.1,
        v_max : float                         = float("inf"),
        x_max : float                         = 5.,
        x_min : float                         = -5.,
    ) -> None:
        self.func    = func
        self.dim     = dim
        self.n       = n
        self.iters   = iters
        self.c0      = c0
        self.c1      = c1
        self.w       = w_max
        self.w_max   = w_max
        self.w_min   = w_min
        self.v_max   = v_max
        self.x_max   = x_max
        self.x_min   = x_min

        self.xs = np.array([x_min + (x_max - x_min) * np.random.rand(self.dim) for _ in range(n)])
        self.x_fits = np.zeros(n)
        self.calc_fitness_vals(self.xs, self.x_fits)

        self.local_best_xs = self.xs.copy()
        self.local_best_fits = self.x_fits.copy()

        self.global_best_x, self.global_best_fit = self.init_global_info()

        self.vs = np.zeros((n, self.dim))

    def run(self, verbose: int = 1) -> tuple[np.ndarray, float]:
        for i in range(self.iters):
            self.step(i, verbose)
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
        print("-" * 50 + f" {i} " + "-" * 50)
        print(f"Inertia weight = {self.w}")
        print(f"Global best fitness = {self.global_best_fit}")

    def print_global_info(self) -> None:
        print(f"Global best point: {[round(float(x), 4) for x in self.global_best_x]}")
        print(f"Global best fitness = {self.global_best_fit}")

    def init_global_info(self) -> tuple[np.ndarray, float]:
        global_min_idx = np.argmin(self.x_fits)
        return self.xs[global_min_idx], self.x_fits[global_min_idx]

    # To be parallelized:
    def calc_fitness_vals(self, xs: np.ndarray, out: np.ndarray) -> None:
        out[:] = np.apply_along_axis(self.func, 1, xs)
    
    # To be parallelized:
    def update_velocities(self) -> None:
        self.vs *= self.w
        self.vs += (
            self.c0 * np.random.rand(*self.xs.shape) * (self.local_best_xs - self.xs) + 
            self.c1 * np.random.rand(*self.xs.shape) * (self.global_best_x[None, :] - self.xs)
        )
        norms = np.linalg.norm(self.vs, axis=1, keepdims=True)
        self.vs = np.where(norms < self.v_max, self.vs, self.vs / norms * self.v_max)

    # To be parallelized:
    def update_positions(self) -> None:
        self.xs += self.vs
        self.xs.clip(self.x_min, self.x_max, out=self.xs)
    
    # To be parallelized:
    def update_bests(self) -> None:
        self.calc_fitness_vals(self.xs, self.x_fits)
        for i, (x, curr_fit, best_fit) in enumerate(zip(self.xs, self.x_fits, self.local_best_fits)):
            if curr_fit < best_fit:
                self.local_best_xs[i] = x.copy()
                self.local_best_fits[i] = curr_fit
                if curr_fit < self.global_best_fit:
                    self.global_best_x = x.copy()
                    self.global_best_fit = curr_fit
    
    def update_inertia_weight(self, i: int) -> None:
        self.w = self.w_max - (self.w_max - self.w_min) * (i / self.iters)


def main():
    seed        = None
    func        = levy_func
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

    pso = PSO(
        func  = func,
        dim   = dim,
        n     = n,
        iters = iters,
        x_min = x_min,
        x_max = x_max,
        v_max = v_max
    )

    if is_make_ani:
        assert dim == 2
        fig, ax, surf, line = plot_func(func, pso.xs, x_min, x_max, markersize=markersize, is_show=False)
        make_animation(pso.step, iters, fig, line, verbose, save_path=f"PSO_{func.__name__}.png")
    else:
        t = timeit.timeit(lambda: pso.run(verbose), number=1)
        print("-" * 100)
        print(f"Total time = {t}")
        if dim == 2: 
            plot_func(func, pso.xs, x_min, x_max, markersize=markersize, is_show=True)
    
    pso.print_global_info()


if __name__ == "__main__":

    main()