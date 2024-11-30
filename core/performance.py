import os
import json
import timeit
import argparse
import warnings

warnings.filterwarnings("ignore")

SAVE_PATH = "assets"


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


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--device", type=str, default="all", help="Whether to use CPU or GPU to run PSO algorithm.", choices=["cpu", "gpu", "all"])
    parser.add_argument("-n", "--name"  , type=str, default="Exp", help="The name of this experiment.")
    parser.add_argument("-l", "--l_ord" , type=int, default=1    , help="The minimal order among the testing dimensions. (dim = 2 ** <order> - 1)")
    parser.add_argument("-r", "--r_ord" , type=int, default=10   , help="The maximal order among the testing dimensions. (dim = 2 ** <order> - 1)")
    parser.add_argument("-i", "--n_iter", type=int, default=10   , help="The number of iterations for running PSOs.")
    parser.add_argument("-t", "--n_test", type=int, default=10   , help="The number of iterations for running Timers.")
    args = parser.parse_args()

    n_ord = args.r_ord - args.l_ord + 1
    dims, cpu_time_alls, cpu_time_mains, gpu_time_alls, gpu_time_mains = [None] * n_ord, [None] * n_ord, [None] * n_ord, [None] * n_ord, [None] * n_ord
    for i in range(args.l_ord, args.r_ord + 1):
        dims[i] = 2 ** i - 1
        print(f"\n >> Iter {i}: (dim={dims[i]})")
        if args.device == "cpu":
            cpu_time_all, cpu_time_main = perf_cpu(dim=dims[i], n_iter=args.n_iter, n_test=args.n_test)
            cpu_time_alls [i] = cpu_time_all  * 1000.
            cpu_time_mains[i] = cpu_time_main * 1000.
        elif args.device == "gpu":
            gpu_time_all, gpu_time_main = perf_gpu(dim=dims[i], n_iter=args.n_iter, n_test=args.n_test)
            gpu_time_alls [i] = gpu_time_all  * 1000.
            gpu_time_mains[i] = gpu_time_main * 1000.
        elif args.device == "all":
            cpu_time_all, cpu_time_main = perf_cpu(dim=dims[i], n_iter=args.n_iter, n_test=args.n_test)
            cpu_time_alls [i] = cpu_time_all  * 1000.
            cpu_time_mains[i] = cpu_time_main * 1000.
            gpu_time_all, gpu_time_main = perf_gpu(dim=dims[i], n_iter=args.n_iter, n_test=args.n_test)
            gpu_time_alls [i] = gpu_time_all  * 1000.
            gpu_time_mains[i] = gpu_time_main * 1000.
        else:
            raise ValueError(f"Invalid device: {args.device}")

    data = {
        "iters"          : [args.n_iter] * n_ord,
        "dims"           : dims,
        "cpu_time_alls"  : cpu_time_alls,
        "cpu_time_mains" : cpu_time_mains,
        "gpu_time_alls"  : gpu_time_alls,
        "gpu_time_mains" : gpu_time_mains,
        "unit"           : "ms"
    }
    
    if not os.path.exists(SAVE_PATH):
        os.makedirs(SAVE_PATH, exist_ok=True)

    with open(f"{SAVE_PATH}/${args.name}_{args.device.upper()}_record.json", "w") as f:
        json.dump(data, f)


if __name__ == "__main__":

    main()