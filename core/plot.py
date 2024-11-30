import os
import json
import argparse
import matplotlib.pyplot as plt

plt.style.use("seaborn-v0_8")

SAVE_PATH = "assets"


def plot_all(args: argparse.Namespace):

    assert not args.rate, "Not support rate figure for all-device comparison."

    datafile = f"{SAVE_PATH}/{args.name}_ALL_record.json"
    with open(datafile, "r") as f:
        data = json.load(f)

    dims           = data["dims"]
    cpu_time_alls  = data["cpu_time_alls"]
    cpu_time_mains = data["cpu_time_mains"]
    gpu_time_alls  = data["gpu_time_alls"]
    gpu_time_mains = data["gpu_time_mains"]

    if not args.all:
        if cpu_time_alls or cpu_time_mains:
            plt.plot(dims, cpu_time_mains, color="blue", linestyle="-", label="CPU main")
        if gpu_time_alls or gpu_time_mains:
            plt.plot(dims, gpu_time_mains, color="green", linestyle="-", label="GPU main")
    else:
        if cpu_time_alls or cpu_time_mains:
            plt.plot(dims, cpu_time_alls, color="blue", linestyle="-", label="CPU all")
            plt.plot(dims, cpu_time_mains, color="cyan", linestyle="--", label="CPU main")
        if gpu_time_alls or gpu_time_mains:
            plt.plot(dims, gpu_time_alls, color="green", linestyle="-", label="GPU all")
            plt.plot(dims, gpu_time_mains, color="lime", linestyle="--", label="GPU main")

    name = f"{args.name}_ALL_times"
    plt.title(name)
    plt.legend()
    plt.xlabel("Dimension")
    plt.ylabel("Time (ms)")
    if args.logx: 
        plt.xscale("log")
        name += "-logx"
    if args.logy: 
        plt.yscale("log")
        name += "-logy"
    plt.savefig(f"{SAVE_PATH}/{name}.png")


def plot_cpu(args: argparse.Namespace):

    datafile = f"{SAVE_PATH}/{args.name}_CPU_record.json"
    if not os.path.exists(datafile):
        datafile = f"{SAVE_PATH}/{args.name}_ALL_record.json"
    with open(datafile, "r") as f:
        data = json.load(f)

    dims           = data["dims"]
    cpu_time_alls  = data["cpu_time_alls"]
    cpu_time_mains = data["cpu_time_mains"]

    if cpu_time_alls or cpu_time_mains:
        if args.rate:
            plt.plot(dims, [ta / tm for ta, tm in zip(cpu_time_alls, cpu_time_mains)], color="red", linestyle="-")
        elif not args.all:
            plt.plot(dims, cpu_time_mains, color="blue", linestyle="-", label="CPU main")
        else:
            plt.plot(dims, cpu_time_alls, color="blue", linestyle="-", label="CPU all")
            plt.plot(dims, cpu_time_mains, color="cyan", linestyle="--", label="CPU main")
    else:
        raise RuntimeError("No CPU experiment data in the file.")

    name = f"{args.name}_CPU_times" if not args.rate else f"{args.name}_Rate_CPU_times"
    plt.title(name)
    plt.legend()
    plt.xlabel("Dimension")
    plt.ylabel("Time (ms)" if not args.rate else "Rate (all/main)")
    if args.logx: 
        plt.xscale("log")
        name += "-logx"
    if args.logy: 
        plt.yscale("log")
        name += "-logy"
    plt.savefig(f"{SAVE_PATH}/{name}.png")


def plot_gpu(args: argparse.Namespace):

    datafile = f"{SAVE_PATH}/{args.name}_GPU_record.json"
    if not os.path.exists(datafile):
        datafile = f"{SAVE_PATH}/{args.name}_ALL_record.json"
    with open(datafile, "r") as f:
        data = json.load(f)

    dims           = data["dims"]
    gpu_time_alls  = data["gpu_time_alls"]
    gpu_time_mains = data["gpu_time_mains"]

    if gpu_time_alls or gpu_time_mains:
        if args.rate:
            plt.plot(dims, [ta / tm for ta, tm in zip(gpu_time_alls, gpu_time_mains)], color="red", linestyle="-")
        elif not args.all:
            plt.plot(dims, gpu_time_mains, color="green", linestyle="-", label="GPU main")
        else:
            plt.plot(dims, gpu_time_alls, color="green", linestyle="-", label="GPU all")
            plt.plot(dims, gpu_time_mains, color="lime", linestyle="--", label="GPU main")
    else:
        raise RuntimeError("No GPU experiment data in the file.")

    name = f"{args.name}_GPU_times" if not args.rate else f"{args.name}_Rate_GPU_times"
    plt.title(name)
    plt.legend()
    plt.xlabel("Dimension")
    plt.ylabel("Time (ms)" if not args.rate else "Rate (all/main)")
    if args.logx: 
        plt.xscale("log")
        name += "-logx"
    if args.logy: 
        plt.yscale("log")
        name += "-logy"
    plt.savefig(f"{SAVE_PATH}/{name}.png")


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--device", type=str, default="all", help="Whether to use CPU or GPU to run PSO algorithm.", choices=["cpu", "gpu", "all"])
    parser.add_argument("-n", "--name"  , type=str, default="Exp", help="The name of this experiment.")
    parser.add_argument("--all" , action="store_true" , help="Also plot setup times.")
    parser.add_argument("--logx", action="store_true" , help="Making x-axis log-scale.")
    parser.add_argument("--logy", action="store_true" , help="Making y-axis log-scale.")
    parser.add_argument("--rate", action="store_true" , help="Making rate comparison figure.")
    args = parser.parse_args()

    if args.device == "cpu":
        plot_cpu(args)
    elif args.device == "gpu":
        plot_gpu(args)
    else:
        plot_all(args)


if __name__ == "__main__":

    main()