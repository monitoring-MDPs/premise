# %%
from argparse import Namespace
import re
from pathlib import Path
from collections import defaultdict
import io
import imageio.v2 as imageio
from matplotlib import pyplot as plt
import numpy as np
from tqdm import tqdm

from premise.interval.interval import analyse_interval_width, create_monitor
from premise.interval.loading import load_imc


def parse_filename(filename):
    """
    Extracts group prefix and iteration number from filename.
    Returns (group, iteration, type) where type is 'interval' or 'initial_interval'.
    """
    m = re.match(r"(.+?)(\d+)-(interval|initial_interval)\.npy$", filename)
    if m:
        prefix = m.group(1)
        iteration = int(m.group(2))
        typ = m.group(3)
        group = prefix
        return group, iteration, typ
    else:
        raise ValueError("Cannot parse filename")


def plot_widths(widths_dict, title=None, typ="errorbar"):
    bar_count = 0
    for l, entries in widths_dict.items():
        if not entries:
            continue
        centers = np.array([e[1] for e in entries])
        widths = np.array([e[0] for e in entries])
        lower = centers - widths / 2
        upper = centers + widths / 2

        if typ == "errorbar":
            plt.errorbar(
                np.arange(bar_count, bar_count + len(entries)),
                centers,
                yerr=widths / 2,
                fmt=".",
                label=f"Length {l}",
            )
        elif typ == "box":
            plt.ylim(0, 1)
            plt.boxplot(
                widths,
                positions=[bar_count + (len(entries) / 2)],
                widths=[len(entries)],
            )
        elif typ == "violin":
            plt.ylim(0, 1)
            plt.violinplot(
                widths,
                positions=[bar_count + (len(entries) / 2)],
                widths=[len(entries)],
            )

        bar_count += len(entries)

    plt.xlim(0, bar_count)

    plt.xlabel("Transition index")

    plt.ylabel("Interval center with width error bars")
    # plt.legend()
    plt.tight_layout()
    if title:
        plt.title(title)


def main(path="../../out/models/2025-06-12_10-31-03", out_path="../../out/tmp/gifs3"):
    out_path = Path(out_path)
    out_path.mkdir(exist_ok=True)
    path = Path(path)
    print(path.exists())
    files = list(path.glob("*.npy"))
    groups = defaultdict(dict)
    plot_type = "violin"

    for file in files:
        try:
            group, iteration, typ = parse_filename(file.name)
            if iteration not in groups[group]:
                groups[group][iteration] = [None, None]

            if typ == "interval":
                groups[group][iteration][1] = file
            else:
                groups[group][iteration][0] = file
        except ValueError:
            print(f"skipping {file.name}")
            continue

    # Load models and collect frames for GIFs
    all_widths = {}
    for group, iters in sorted(groups.items()):
        print(f"Analysing {group}")
        all_widths[group] = {}
        frames = []
        for iteration, (init_path, trans_path) in tqdm(sorted(iters.items())):
            interval, init_interval = load_imc(
                Namespace(init_path=str(init_path), trans_path=str(trans_path))
            )

            mon, mon_comps = create_monitor(interval, init_interval, "min", "crash", 15)

            widths = analyse_interval_width(mon_comps.ipomdp)
            all_widths[group][iteration] = widths

            plt.figure(figsize=(20, 6))
            plot_widths(widths, title=f"Widths of {group}:{iteration}", typ=plot_type)
            buf = io.BytesIO()
            plt.savefig(buf, format="png")
            plt.close()
            buf.seek(0)
            frames.append(imageio.imread(buf))
            buf.close()

        # Save GIF for this group
        if frames:
            gif_path = out_path / f"{group}_{plot_type}_widths.gif"
            imageio.mimwrite(gif_path, frames, duration=100, loop=0)
            print(f"Saved GIF for {group} at {gif_path}")
            # display(Video(filename=str(gif_path)))

    return all_widths


all_widths = main()

# %%
