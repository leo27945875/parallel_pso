import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

plt.style.use("seaborn-v0_8")

VALID_DEVICES  = ["omp", "pthread"]
BASE_SAVE_PATH = "assets"
DEVICE_COLOR_MAP = {
    "cpu": ["blue", "cyan"],
    "gpu": ["green", "lime"],
    "omp": ["red", "lightcoral"],
    "pthread": ["purple", "violet"]
}


def plot_(args: argparse.Namespace) -> None:
    datafile = f"{BASE_SAVE_PATH}/{args.name}/{args.name}_{args.device.upper()}_scaling.json"
    with open(datafile, 'r') as f:
        data = json.load(f)

    xoffset = 15
    if args.logx:
        xoffset *= 5

    dims = data["dims"]
    colors = plt.cm.jet(np.linspace(0, 1, len(data["perf_map"])))
    for i, (nthd, times) in enumerate(data["perf_map"].items()):
        plt.plot(dims, times, "-o", color=colors[i], label=f"{args.device}_threads={nthd}")
        plt.text(dims[-1] + xoffset, times[-1], f"{round(times[-1])}", fontsize=15, va="center", ha="left")
    
    name = f"{args.device.upper()}_Scaling_Curves"

    plt.title(f"{args.device.upper()} Scaling Curves")
    plt.legend()
    plt.xlabel("Dimension")
    plt.ylabel("Time (ms)")
    if args.logx:
        plt.xscale("log")
        name += "-logx"
    if args.logy:
        plt.yscale("log")
        name += "-logy"
    plt.gca().xaxis.set_major_formatter(ScalarFormatter())
    plt.xticks(dims)

    save_folder = f"{BASE_SAVE_PATH}/{args.name}"
    os.makedirs(save_folder, exist_ok=True)
    plt.savefig(f"{save_folder}/{name}.png")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--name"  , type=str, default="Scl", help="The name of this experiment.")
    parser.add_argument("-d", "--device", type=str, default="omp", help="What device to run PSO algorithm.", choices=VALID_DEVICES)
    parser.add_argument("--logx", action="store_true", help="Making x-axis log-scale.")
    parser.add_argument("--logy", action="store_true", help="Making y-axis log-scale.")
    args = parser.parse_args()
    plot_(args)


if __name__ == "__main__":

    main()