import numpy as np


def levy_func(x: np.ndarray) -> float:
    n = x.shape[0]
    w = lambda z: 1. + 0.25 * (z - 1.)
    return (
        np.sin(np.pi * w(x[0])) ** 2 + 
        sum((w(x[i]) - 1.) ** 2 * (1. + 10. * np.sin(np.pi * w(x[i]) + 1.) ** 2) for i in range(n - 1)) + 
        (w(x[n-1]) - 1.) ** 2 * (1. + np.sin(2. * np.pi * w(x[n-1])) ** 2)
    )


def rastrigin_func(x: np.ndarray) -> float:
    n = x.shape[0]
    A = 100.
    return A * n + sum(x[i] ** 2  - A * np.cos(0.5 * np.pi * x[i]) for i in range(n))


def rosenbrock_func(x: np.ndarray) -> float:
    n = x.shape[0]
    return sum(2. * (x[i+1] - x[i] ** 2) ** 2 + (1. - x[i]) ** 2 for i in range(n - 1))


if __name__ == "__main__":

    print(
        levy_func(np.array([1., 1.]))
    )
    print(
        levy_func(np.array([-1., -1.]))
    )
    print(
        levy_func(np.array([10., 5.]))
    )