import os
import matplotlib.pyplot as plt
import numpy as np
import argparse
import warnings

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--load_folder", type=str, help="folder for loading the npys")
    parser.add_argument("--save_file_name", default="temp", type=str, help="file name for saving the pdf")
    parser.add_argument("--load_feats", type=str, help="files for loading the plot results")
    parser.add_argument("--names", type=str, help="names for the plot results")
    parser.add_argument("--title", type=str, help="the plot title")

    args = parser.parse_args()
    files = eval(args.load_feats)  # list[str]
    names = eval(args.names)  # list[str]
    assert len(files) == len(names)

    fig, ax = plt.subplots()
    # ax.set_title(args.title)
    ax.set_xlabel('Certified Radius')
    ax.set_ylabel('Certified Accuracy')
    x_max = 0
    i = 0
    al = 1
    for file in files:
        npy_file = os.path.join(args.load_folder, f"{file}.npy")
        if os.path.exists(npy_file):
            res = np.load(npy_file, allow_pickle=True)
            x = list(np.unique(res))
            y = []
            p = 0
            for j in range(len(x)):
                while res[p] != x[j]: p += 1
                y.append((len(res) - p) * 100.0 / len(res))
            x.append(x[-1] + 1)
            y.append(0)

            if x[0] < -0.5:
                x = x[1:]
                y = y[1:]
            ax.plot(x, y, color="k", label=names[i], alpha=al)
            al *= 0.6
            # ax.axhline(y=y[0], color=colors[i], linestyle='dashed', linewidth=0.5)
            i += 1
            x_max = max(x_max, x[-1])
        else:
            warnings.warn(f"{npy_file} does not detected!")

    ax.set(xlim=(0, x_max + 5), ylim=(0, 100))
    ax.legend()
    plt.show()
    plt.savefig(f"./{args.load_folder}/{args.save_file_name}.pdf")
