import numpy as np
from typing import Callable, Iterable

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import cm
from matplotlib.figure import Figure
from matplotlib.artist import Artist


def calc_plot_points(points: np.ndarray, func: Callable[[np.ndarray], float]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xp = points[:, 0]
    yp = points[:, 1]
    zp = np.apply_along_axis(func, 1, points)
    return xp, yp, zp


def plot_func(func: Callable[[np.ndarray], float], points: np.ndarray | None = None, x_min: float = -5., x_max: float = 5., step: float = 0.1, markersize: int = 3, is_contour: bool = False, is_show: bool = True) -> None:
    X = np.arange(x_min, x_max, step)
    Y = np.arange(x_min, x_max, step)
    X, Y = np.meshgrid(X, Y)
    Z = np.apply_along_axis(func, 0, np.stack([X, Y], axis=0))

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    ax.zaxis.set_major_formatter('{x:.02f}')

    surf = ax.plot_surface(X, Y, Z, edgecolor='royalblue', lw=0.1, rstride=8, cstride=8, alpha=0.3, cmap=cm.coolwarm)
    fig.colorbar(surf, aspect=5)
    if points is not None:
        xp, yp, zp = calc_plot_points(points, func)
        line = ax.plot(xp, yp, zp, 'o', c="r", markersize=markersize)[0]
    else:
        line = None

    if is_contour: 
        ax.contourf(X, Y, Z, zdir='z', offset=0, cmap='coolwarm')

    if is_show:
        plt.show()
    
    return fig, ax, surf, line


def make_animation(animate_func: Callable[[int], Iterable[Artist]], iters: int, fig: Figure, line: Artist, verbose: int = 1, save_path: str = "assets/PSO.gif", fps: int = 15) -> None:
    ani = animation.FuncAnimation(fig, animate_func, iters, fargs=(verbose, line), interval=100)
    writer = animation.PillowWriter(fps=fps, metadata=dict(artist='Me'), bitrate=1800)
    ani.save(save_path, writer=writer)