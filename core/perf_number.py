import os
import json
import timeit
import argparse
import warnings

warnings.filterwarnings("ignore")

VALID_DEVICES   = ["gpu", "omp", "pthread", "cpu"]
BASE_SAVE_PATH  = "assets"
TARGET_FUNCTION = "levy_func"
FIX_NUM         = 3000


def exec_timer(timer: timeit.Timer, prefix: str, n_test: int, is_all: bool) -> tuple[float, float]:
    print("=" * 50 + f" {prefix} " + "=" * 50)
    if is_all:
        print(f"{prefix} all time = ", end="")
        time_all = sum(timer.repeat(n_test, number=1)) / n_test
        print(f"{time_all}(s)")
    else:
        time_all = None

    print(f"{prefix} main time = ", end="")
    time_main = timer.timeit(n_test) / n_test
    print(f"{time_main}(s)")
    return time_all, time_main


def perf_cpu(
    dim  : int,
    num  : int,
    args : argparse.Namespace
) -> tuple[float, float]:

    cpu_setup_str = \
f"""
import numpy as np

from core.funcs import {TARGET_FUNCTION}
from core.pypso import PSO

x_min   = -20
x_max   = 20.
v_max   = 5.
verbose = 0

pso = PSO(
    func  = {TARGET_FUNCTION}, 
    dim   = {dim},
    n     = {num},
    iters = {args.n_iter},
    x_min = x_min,
    x_max = x_max,
    v_max = v_max
)

    """

    cpu_stmt_str = \
"""
pso.run(verbose)
"""

    cpu_timer = timeit.Timer(cpu_stmt_str, cpu_setup_str)
    cpu_time_all, cpu_time_main = exec_timer(cpu_timer, "CPU", args.n_test, args.all)
    return cpu_time_all, cpu_time_main


def perf_omp(
    dim  : int,
    num  : int,
    args : argparse.Namespace
) -> tuple[float, float]:

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
    n        = {num},
    iters    = {args.n_iter},
    n_thread = {args.n_core},
    x_min    = x_min,
    x_max    = x_max,
    v_max    = v_max
)

    """

    omp_stmt_str = \
"""
pso.run(verbose)
"""

    omp_timer = timeit.Timer(omp_stmt_str, omp_setup_str)
    omp_time_all, omp_time_main = exec_timer(omp_timer, "OMP", args.n_test, args.all)
    return omp_time_all, omp_time_main


def perf_pthread(
    dim  : int,
    num  : int,
    args : argparse.Namespace
) -> tuple[float, float]:

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
    n        = {num},
    iters    = {args.n_iter},
    n_thread = {args.n_core},
    x_min    = x_min,
    x_max    = x_max,
    v_max    = v_max
)

    """

    pthread_stmt_str = \
"""
pso.run(verbose)
"""

    pthread_timer = timeit.Timer(pthread_stmt_str, pthread_setup_str)
    pthread_time_all, pthread_time_main = exec_timer(pthread_timer, "Pthread", args.n_test, args.all)
    return pthread_time_all, pthread_time_main


def perf_gpu(
    dim  : int,
    num  : int,
    args : argparse.Namespace
) -> tuple[float, float]:

    cuda_setup_str = \
f"""
from core.pycupso import PSO_CUDA

x_min   = -20
x_max   = 20.
v_max   = 5.
verbose = 0

pso = PSO_CUDA(
    dim   = {dim},
    n     = {num},
    iters = {args.n_iter},
    x_min = x_min,
    x_max = x_max,
    v_max = v_max
)

    """

    cuda_stmt_str = \
"""
pso.run(verbose)
"""

    cuda_timer = timeit.Timer(cuda_stmt_str, cuda_setup_str)
    cuda_time_all, cuda_time_main = exec_timer(cuda_timer, "CUDA", args.n_test, args.all)
    return cuda_time_all, cuda_time_main


def perf_one_device(args: argparse.Namespace, device: str | None = None) -> None:
    if not device:
        device = args.device

    n_ord = args.r_ord - args.l_ord + 1
    nums, time_alls, time_mains = [None] * n_ord, [None] * n_ord, [None] * n_ord
    for i, o in enumerate(range(args.l_ord, args.r_ord + 1)):
        nums[i] = 2 ** o - 1
        if args.type == "dim":
            d, n = nums[i], FIX_NUM
        else:
            d, n = FIX_NUM, nums[i]
        print(f"\n >> Iter {i + 1}: (dim={d}, num={n})")
        time_all, time_main = eval(f"perf_{device}")(d, n, args)
        time_alls [i] = time_all  * 1000. if time_all is not None else None
        time_mains[i] = time_main * 1000.
        print("=" * (102 + len(device)))

    data = {
        "iters"      : [args.n_iter] * n_ord,
        "dims"       : nums,
        "time_alls"  : time_alls,
        "time_mains" : time_mains,
        "unit"       : "ms"
    }
    
    save_path = f"{BASE_SAVE_PATH}/{args.name}"
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)

    device_name = f"{device.upper()}{'_threads=' + str(args.n_core) if device in ('omp', 'pthread') else ''}"
    with open(f"{save_path}/{args.name}-{args.type}_{device_name}.json", "w") as f:
        json.dump(data, f)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--name"  , type=str, default="Exp", help="The name of this experiment.")
    parser.add_argument("-d", "--device", type=str, default="cpu", help="What device to run PSO algorithm.", choices=[*VALID_DEVICES, "every"])
    parser.add_argument("-t", "--type"  , type=str, default="dim", help="Which axis to scale up.", choices=["dim", "num"])
    parser.add_argument("-l", "--l_ord" , type=int, default=1    , help="The minimal order among the testing numbers. (dim = 2 ** <order> - 1)")
    parser.add_argument("-r", "--r_ord" , type=int, default=10   , help="The maximal order among the testing numbers. (dim = 2 ** <order> - 1)")
    parser.add_argument("-i", "--n_iter", type=int, default=10   , help="The number of iterations for running PSOs.")
    parser.add_argument("-c", "--n_core", type=int, default=16   , help="The number of threads for running OpenMP & pthread implementations.")
    parser.add_argument(      "--n_test", type=int, default=10   , help="The number of iterations for running Timers.")
    parser.add_argument(      "--all"   , action="store_true"    , help="Also plot setup times.")
    args = parser.parse_args()

    if args.device == "every":
        for device in VALID_DEVICES:
            perf_one_device(args, device)
    else:
        perf_one_device(args)


if __name__ == "__main__":

    main()