import os
import json
import timeit
import argparse
import warnings
from collections import defaultdict

warnings.filterwarnings("ignore")

VALID_DEVICES   = ["omp", "pthread"]
BASE_SAVE_PATH  = "assets"
TARGET_FUNCTION = "levy_func"


def exec_timer(timer: timeit.Timer, prefix: str, n_test: int) -> float:
    print("=" * 50 + f" {prefix} " + "=" * 50)
    print(f"{prefix} main time = ", end="")
    time_main = timer.timeit(n_test) / n_test
    print(f"{time_main}(s)")
    return time_main


def scale_omp(dim: int, n_thread: int, n_test: int, n_iter: int) -> float:
    
    omp_setup_str = \
f"""
import numpy as np

from core.funcs import {TARGET_FUNCTION}
from core.pyomppso import PSO_OMP

x_min   = -20
x_max   = 20.
v_max   = 5.
verbose = 0

pso = PSO_OMP(
    func     = {TARGET_FUNCTION}, 
    dim      = {dim},
    n        = {dim * 2**5},
    iters    = {n_iter},
    n_thread = {n_thread},
    x_min    = x_min,
    x_max    = x_max,
    v_max    = v_max
)

    """

    omp_stmt_str = \
"""
pso.run(verbose)
"""

    timer = timeit.Timer(omp_stmt_str, omp_setup_str)
    return exec_timer(timer, "OMP", n_test)


def scale_pthread(dim: int, n_thread: int, n_test: int, n_iter: int) -> float:
    
    pthread_setup_str = \
f"""
import numpy as np

from core.funcs import {TARGET_FUNCTION}
from core.pypthreadpso import PSO_Pthread

x_min   = -20
x_max   = 20.
v_max   = 5.
verbose = 0

pso = PSO_Pthread(
    func     = {TARGET_FUNCTION}, 
    dim      = {dim},
    n        = {dim * 2**5},
    iters    = {n_iter},
    n_thread = {n_thread},
    x_min    = x_min,
    x_max    = x_max,
    v_max    = v_max
)

    """

    pthread_stmt_str = \
"""
pso.run(verbose)
"""

    timer = timeit.Timer(pthread_stmt_str, pthread_setup_str)
    return exec_timer(timer, "Pthread", n_test)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--name"  , type=str, default="Scl", help="The name of this experiment.")
    parser.add_argument("-d", "--device", type=str, default="omp", help="What device to run PSO algorithm.", choices=VALID_DEVICES)
    parser.add_argument("-l", "--l_nthd", type=int, default=0    , help="The minimal order of threads.")
    parser.add_argument("-r", "--r_nthd", type=int, default=4    , help="The minimal order of threads.")
    parser.add_argument("-s", "--s_nthd", type=int, default=1    , help="The step of order of threads.")
    parser.add_argument("-i", "--n_iter", type=int, default=10   , help="The number of iterations for running PSOs.")
    parser.add_argument("-t", "--n_test", type=int, default=10   , help="The number of iterations for running Timers.")
    parser.add_argument("--l_ord", type=int, default=1 , help="The minimal order among the testing dimensions. (dim = 2 ** <order> - 1)")
    parser.add_argument("--r_ord", type=int, default=10, help="The maximal order among the testing dimensions. (dim = 2 ** <order> - 1)")
    args = parser.parse_args()


    dims = [2 ** o - 1 for o in range(args.l_ord, args.r_ord + 1)]
    nthds = [2 ** o for o in range(args.l_nthd, args.r_nthd + 1, args.s_nthd)]

    perf_map = defaultdict(list)
    for nthd in nthds:
        for dim in dims:
            print(f"\n >> N_Threads {nthd}: (dim={dim}, num={dim * 2**5})")
            time_main = eval(f"scale_{args.device}")(dim, nthd, args.n_test, args.n_iter) * 1000.
            perf_map[nthd].append(time_main)
            print("=" * (102 + len(args.device)))

    save_path = f"{BASE_SAVE_PATH}/{args.name}"
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)

    data = {
        "dims": dims, "perf_map": dict(perf_map)
    }
    with open(f"{save_path}/{args.name}_{args.device.upper()}_scaling.json", 'w') as f:
        json.dump(data, f)


if __name__ == "__main__":

    main()