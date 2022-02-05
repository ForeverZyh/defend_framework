import os
import matplotlib.pyplot as plt
import numpy as np
import argparse
import warnings

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--load_dir", type=str, help="parent dir for loading the folders")
    parser.add_argument("--save_file_name", default="temp", type=str, help="file name for saving the pdf")
    parser.add_argument("--load_folders", type=str, help="folders for loading the plot results")
    parser.add_argument("--poisoned_feat_num", default=None, type=int,
                        help="the poisoned feature number. None means all poisoned")

    args = parser.parse_args()
    folders = eval(args.load_folders)  # list[str]

    fig, ax = plt.subplots()
    ax.set_title("No Constrain on the Poisoned Feature Number" if args.poisoned_feat_num is None else
                 f"Poisoned Feature Number = {args.poisoned_feat_num}")
    ax.set_xlabel('Poisoned Instance Number')
    ax.set_ylabel('Certified Accuracy')
    x_max = 0
    for folder_ in folders:
        folder = os.path.join(args.load_dir, folder_)
        if os.path.exists(os.path.join(folder, f"plot_{args.poisoned_feat_num}.npy")):
            res = dict(np.load(os.path.join(folder, f"plot_{args.poisoned_feat_num}.npy"), allow_pickle=True).item())
            x = sorted(list(res.keys()))
            x_max = max(x_max, max(x))
        else:
            warnings.warn(f"{os.path.join(folder, f'plot_{args.poisoned_feat_num}.npy')} does not detected!")

    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']
    for i, folder_ in enumerate(folders):
        folder = os.path.join(args.load_dir, folder_)
        if os.path.exists(os.path.join(folder, f"plot_{args.poisoned_feat_num}.npy")):
            res = dict(np.load(os.path.join(folder, f"plot_{args.poisoned_feat_num}.npy"), allow_pickle=True).item())
            x = sorted(list(res.keys()))
            y = [np.mean(res[i] == 1) * 100 for i in x]
            if x[-1] != x_max:
                x.append(x_max)
                y.append(0)
            ax.plot(x, y, label=folder_, color=colors[i])
            ax.plot(x, [y[0]] * len(x), color=colors[i], linestyle='dashed', linewidth=0.5)


    ax.set(xlim=(0, x_max), ylim=(0, 100))
    ax.legend()
    plt.show()
    plt.savefig(f"./res/{args.save_file_name}.pdf")
