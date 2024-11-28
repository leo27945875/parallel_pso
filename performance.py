import json
import timeit
import argparse
import warnings

warnings.filterwarnings("ignore")


def exec_timer(timer: timeit.Timer, prefix: str, n_test: int = 5) -> tuple[float, float]:
    print("\n" + "=" * 50 + f" {prefix} " + "=" * 50)
    print(f"{prefix} all time = ", end="")
    time_all = sum(timer.repeat(n_test, number=1)) / n_test
    print(f"{time_all}(s)")
    print(f"{prefix} main time = ", end="")
    time_main = timer.timeit(n_test) / n_test
    print(f"{time_main}(s)")
    print("=" * (102 + len(prefix)))
    return time_all, time_main


def perf_cpu(
    dim   : int = 64,
    iters : int = 10,
    func  : str = "levy_func"
) -> tuple[float, float]:

    cpu_setup_str = \
f"""
import numpy as np

from funcs import {func}
from pypso import PSO

x_min   = -20
x_max   = 20.
v_max   = 5.
verbose = 0

pso = PSO(
    func   = {func}, 
    dim    = {dim},
    n      = {dim * 2**5},
    iters  = {iters},
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
    cpu_time_all, cpu_time_main = exec_timer(cpu_timer, "CPU")
    return cpu_time_all, cpu_time_main


def perf_gpu(
    dim   : int = 64, 
    iters : int = 10
) -> tuple[float, float]:

    cuda_setup_str = \
f"""
from pycupso import PSO_CUDA

x_min   = -20
x_max   = 20.
v_max   = 5.
verbose = 0

pso = PSO_CUDA(
    dim    = {dim},
    n      = {dim * 2**5},
    iters  = {iters},
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
    cuda_time_all, cuda_time_main = exec_timer(cuda_timer, "CUDA")
    return cuda_time_all, cuda_time_main


def main():

    n_ord  = 10
    n_iter = 10

    parser = argparse.ArgumentParser()
    parser.add_argument("device", type=str, choices=["cpu", "gpu", "all"], help="Whether to use CPU or GPU to run PSO algorithm.")
    args = parser.parse_args()

    dims, cpu_time_alls, cpu_time_mains, gpu_time_alls, gpu_time_mains = [None] * n_ord, [None] * n_ord, [None] * n_ord, [None] * n_ord, [None] * n_ord
    for i in range(n_ord):
        print(f"Iter {i}:")
        dims[i] = 3 * 2**i
        if args.device == "cpu":
            cpu_time_all, cpu_time_main = perf_cpu(dim=dims[i], iters=n_iter)
            cpu_time_alls [i] = cpu_time_all * 1000.
            cpu_time_mains[i] = cpu_time_main * 1000.
        elif args.device == "gpu":
            gpu_time_all, gpu_time_main = perf_gpu(dim=dims[i], iters=n_iter)
            gpu_time_alls [i] = gpu_time_all * 1000.
            gpu_time_mains[i] = gpu_time_main * 1000.
        elif args.device == "all":
            cpu_time_all, cpu_time_main = perf_cpu(dim=dims[i], iters=n_iter)
            cpu_time_alls [i] = cpu_time_all * 1000.
            cpu_time_mains[i] = cpu_time_main * 1000.
            gpu_time_all, gpu_time_main = perf_gpu(dim=dims[i], iters=n_iter)
            gpu_time_alls [i] = gpu_time_all * 1000.
            gpu_time_mains[i] = gpu_time_main * 1000.
        else:
            raise ValueError(f"Invalid device: {args.device}")

    data = {
        "iters"          : [n_iter] * n_ord,
        "dims"           : dims,
        "cpu_time_alls"  : cpu_time_alls,
        "cpu_time_mains" : cpu_time_mains,
        "gpu_time_alls"  : gpu_time_alls,
        "gpu_time_mains" : gpu_time_mains,
        "unit"           : "ms"
    }
    with open(f"assets/Comp_{args.device.upper()}_record.json", "w") as f:
        json.dump(data, f)


if __name__ == "__main__":

    main()