import os
import json
import argparse
import warnings

from .performance import perf_cpu, perf_gpu

warnings.filterwarnings("ignore")


SAVE_PATH = "assets"


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