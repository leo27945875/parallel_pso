import timeit


def exec_timer(timer: timeit.Timer, prefix: str, n_test: int) -> tuple[float, float]:
    print("=" * 50 + f" {prefix} " + "=" * 50)
    print(f"{prefix} all time = ", end="")
    time_all = sum(timer.repeat(n_test, number=1)) / n_test
    print(f"{time_all}(s)")
    print(f"{prefix} main time = ", end="")
    time_main = timer.timeit(n_test) / n_test
    print(f"{time_main}(s)")
    print("=" * (102 + len(prefix)))
    return time_all, time_main


def perf_cpu(
    dim    : int,
    n_iter : int,
    n_test : int,
    func   : str = "levy_func"
) -> tuple[float, float]:

    cpu_setup_str = \
f"""
import numpy as np

from core.funcs import {func}
from core.pypso import PSO

x_min   = -20
x_max   = 20.
v_max   = 5.
verbose = 0

pso = PSO(
    func   = {func}, 
    dim    = {dim},
    n      = {dim * 2**5},
    iters  = {n_iter},
    x_min  = x_min,
    x_max  = x_max,
    v_max  = v_max
)

    """

    cpu_stmt_str = \
"""
pso.run(verbose)
"""

    cpu_timer = timeit.Timer(cpu_stmt_str, cpu_setup_str)
    cpu_time_all, cpu_time_main = exec_timer(cpu_timer, "CPU", n_test)
    return cpu_time_all, cpu_time_main


def perf_gpu(
    dim    : int, 
    n_iter : int,
    n_test : int
) -> tuple[float, float]:

    cuda_setup_str = \
f"""
from core.pycupso import PSO_CUDA

x_min   = -20
x_max   = 20.
v_max   = 5.
verbose = 0

pso = PSO_CUDA(
    dim    = {dim},
    n      = {dim * 2**5},
    iters  = {n_iter},
    x_min  = x_min,
    x_max  = x_max,
    v_max  = v_max
)

    """

    cuda_stmt_str = \
"""
pso.run(verbose)
"""

    cuda_timer = timeit.Timer(cuda_stmt_str, cuda_setup_str)
    cuda_time_all, cuda_time_main = exec_timer(cuda_timer, "CUDA", n_test)
    return cuda_time_all, cuda_time_main