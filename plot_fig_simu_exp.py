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
    parser.add_argument("--names", type=str, help="names for the plot results")
    parser.add_argument("--title", type=str, help="the plot title")
    parser.add_argument("--poisoned_feat_num", default=None, type=int,
                        help="the poisoned feature number. None means all poisoned")
    parser.add_argument("--imprecise", action='store_true', help="draw imprecise figures (legacy)")

    args = parser.parse_args()
    folders = eval(args.load_folders)  # list[str]
    names = eval(args.names)  # list[str]
    assert len(folders) == len(names)

    fig, ax = plt.subplots()
    # ax.set_title(args.title)
    ax.set_xlabel('Certified Radius')
    ax.set_ylabel('Certified Accuracy')
    if args.imprecise:
        x_max = 0
        for folder_ in folders:
            folder = os.path.join(args.load_dir, folder_)
            if os.path.exists(os.path.join(folder, f"plot_{args.poisoned_feat_num}.npy")):
                res = dict(
                    np.load(os.path.join(folder, f"plot_{args.poisoned_feat_num}.npy"), allow_pickle=True).item())
                x = sorted(list(res.keys()))
                auc = 0
                mcr = 0
                for i in range(len(x)):
                    if i != len(x) - 1:
                        y1 = np.mean(res[x[i]] > 0)
                        y2 = np.mean(res[x[i + 1]] > 0)
                        auc += (x[i + 1] - x[i]) * (y1 + y2) / 2
                        if y1 >= 0.5 and y2 <= 0.5:
                            mcr = (x[i] * (0.5 - y2) + x[i + 1] * (y1 - 0.5)) / (y1 - y2)
                print(f"{folder}\tNormal Acc: {np.mean(res[0] > 0) * 100:.2f}\tAUC: {auc:.2f}\tMCR: {mcr:.2f}")
                x_max = max(x_max, max(x))
            else:
                warnings.warn(f"{os.path.join(folder, f'plot_{args.poisoned_feat_num}.npy')} does not detected!")

        warnings.warn("The above stats are approximated!")
        prop_cycle = plt.rcParams['axes.prop_cycle']
        colors = prop_cycle.by_key()['color']
        for i, folder_ in enumerate(folders):
            folder = os.path.join(args.load_dir, folder_)
            if os.path.exists(os.path.join(folder, f"plot_{args.poisoned_feat_num}.npy")):
                res = dict(
                    np.load(os.path.join(folder, f"plot_{args.poisoned_feat_num}.npy"), allow_pickle=True).item())
                x = sorted(list(res.keys()))
                y = [np.mean(res[i] == 1) * 100 for i in x]
                if x[-1] != x_max:
                    x.append(x_max)
                    y.append(0)
                ax.plot(x, y, color=colors[i], label=names[i])
                ax.axhline(y=y[0], color=colors[i], linestyle='dashed', linewidth=0.5)
    else:
        x_max = 0
        prop_cycle = plt.rcParams['axes.prop_cycle']
        colors = prop_cycle.by_key()['color']
        i = 0
        for folder_ in folders:
            folder = os.path.join(args.load_dir, folder_)
            if os.path.exists(os.path.join(folder, f"plot_{args.poisoned_feat_num}.npy")):
                res = np.load(os.path.join(folder, f"plot_{args.poisoned_feat_num}.npy"), allow_pickle=True)
                x = list(np.unique(res))
                y = []
                p = 0
                for j in range(len(x)):
                    while res[p] != x[j]: p += 1
                    y.append((len(res) - p) * 100.0 / len(res))
                x.append(x[-1] + 1)
                y.append(0)

                for j in range(len(x)):
                    if x[j] < -1:
                        continue
                    else:
                        dash_y = y[j]
                        break

                while x[0] < -0.5:
                    x = x[1:]
                    y = y[1:]
                if x[0] != 0:
                    y = [y[0]] + y
                    x = [0] + x
                ax.plot(x, y, color=colors[i], label=names[i])
                ax.axhline(y=dash_y, color=colors[i], linestyle='dashed', linewidth=0.5)
                i += 1
                x_max = max(x_max, x[-1])
            else:
                warnings.warn(f"{os.path.join(folder, f'plot_{args.poisoned_feat_num}.npy')} does not detected!")

    ax.set(xlim=(0, x_max + 5), ylim=(0, 100))
    ax.legend()
    plt.show()
    plt.savefig(f"./{args.load_dir}/{args.save_file_name}.pdf")
