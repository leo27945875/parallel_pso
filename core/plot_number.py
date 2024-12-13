import os
import json
import argparse
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.ticker import ScalarFormatter

plt.style.use("seaborn-v0_8")

VALID_DEVICES  = ["gpu", "omp", "pthread", "cpu"]
BASE_SAVE_PATH = "assets"
DEVICE_COLOR_MAP = {
    "cpu": ["blue", "cyan"],
    "gpu": ["green", "lime"],
    "omp": ["red", "lightcoral"],
    "pthread": ["purple", "violet"]
}


def plot_(ax: Axes, name: str, device: str, is_all: bool, is_rate: bool, all_color: str, main_color: str, args: argparse.Namespace) -> tuple[str, dict]:
    device_name = f"{device.upper()}{'_threads=' + str(args.n_core) if device in ('omp', 'pthread') else ''}"
    datafile = f"{BASE_SAVE_PATH}/{name}/{name}-{args.type}_{device_name}.json"
    try:
        with open(datafile, "r") as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"[warning] Didn't find file '{datafile}' !")
        return None, None

    dims       = data["dims"]
    time_alls  = data["time_alls"]
    time_mains = data["time_mains"]

    xoffset = 15
    if args.logx:
        xoffset *= 5
    
    if time_alls or time_mains:
        if is_rate:
            ax.plot(dims, [ta / tm for ta, tm in zip(time_alls, time_mains)], color=all_color, linestyle="-")
        elif not is_all:
            ax.plot(dims, time_mains, color=all_color, marker="o", linestyle="-", label=f"{device_name.lower()}")
            ax.text(dims[-1] + xoffset, time_mains[-1], f"{round(time_mains[-1])}", fontsize=15, va="center", ha="left")
        else:
            ax.plot(dims, time_alls, color=all_color, marker="o", linestyle="-", label=f"{device_name.lower()} all")
            ax.plot(dims, time_mains, color=main_color, marker="o", linestyle="--", label=f"{device_name.lower()} main")
    else:
        raise RuntimeError(f"No {device.upper()} experiment data in the file.")
    
    return device_name, data


def plot_one_device(args: argparse.Namespace):

    fig, ax = plt.subplots(1, 1)
    device_name, data = plot_(ax, args.name, args.device, args.all, args.rate, *DEVICE_COLOR_MAP[args.device], args)

    name = f"{args.name}-{args.type}_{device_name}_Times" if not args.rate else f"{args.name}-{args.type}_{device_name}_Rates"
    ax.set_title(f"{args.name} {device_name} Times (scale: {args.type})" if not args.rate else f"{args.name} {device_name} Rates (scale: {args.type})")
    ax.legend()
    ax.set_xlabel("Dimension" if args.type == "dim" else "Number")
    ax.set_ylabel("Time (ms)" if not args.rate else "Rate (all/main)")
    if args.logx: 
        ax.set_xscale("log")
        name += "-logx"
    if args.logy: 
        ax.set_yscale("log")
        name += "-logy"
    ax.set_xticks(data["dims"])
    ax.xaxis.set_major_formatter(ScalarFormatter())
    
    save_folder = f"{BASE_SAVE_PATH}/{args.name}"
    os.makedirs(save_folder, exist_ok=True)
    fig.savefig(f"{save_folder}/{name}.png")


def plot_all_devices(args: argparse.Namespace):

    assert not args.rate, "Not support rate figure for both-device comparison."

    fig, ax = plt.subplots(1, 1)
    data = None
    for device in VALID_DEVICES:
        if device == "cpu" and not args.cpu: continue
        tmp = plot_(ax, args.name, device, args.all, False, *DEVICE_COLOR_MAP[device], args)[1]
        if tmp:
            data = tmp

    name = f"{args.name}-{args.type}_ALL_Times"
    ax.set_title(f"{args.name} ALL Times (scale: {args.type})")
    ax.legend()
    ax.set_xlabel("Dimension" if args.type == "dim" else "Number")
    ax.set_ylabel("Time (ms)")
    if args.logx: 
        ax.set_xscale("log")
        name += "-logx"
    if args.logy: 
        ax.set_yscale("log")
        name += "-logy"
    ax.set_xticks(data["dims"])
    ax.xaxis.set_major_formatter(ScalarFormatter())

    save_folder = f"{BASE_SAVE_PATH}/{args.name}"
    os.makedirs(save_folder, exist_ok=True)
    fig.savefig(f"{save_folder}/{name}.png")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--name"  , type=str, default="Exp", help="The name of this experiment.")
    parser.add_argument("-d", "--device", type=str, default="cpu", help="What device to run PSO algorithm.", choices=[*VALID_DEVICES, "every"])
    parser.add_argument("-t", "--type"  , type=str, default="dim", help="Which axis to scale up.", choices=["dim", "num"])
    parser.add_argument("-c", "--n_core", type=int, default=16   , help="The number of threads for running OpenMP & pthread implementations.")
    parser.add_argument("--all" , action="store_true", help="Also plot setup times.")
    parser.add_argument("--logx", action="store_true", help="Making x-axis log-scale.")
    parser.add_argument("--logy", action="store_true", help="Making y-axis log-scale.")
    parser.add_argument("--rate", action="store_true", help="Making rate comparison figure.")
    parser.add_argument("--cpu" , action="store_true", help="Whether to plot CPU curve.")
    args = parser.parse_args()

    if args.device == "every":
        plot_all_devices(args)
    else:
        plot_one_device(args)


if __name__ == "__main__":

    main()