import os
import matplotlib.pyplot as plt
import numpy as np
import argparse
import warnings

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--load_dir", type=str, help="parent dir for loading the folders")
    parser.add_argument("--save_file_name", default="temp", type=str, help="file name for saving the pdf")
    parser.add_argument("--load_folder", type=str, help="folder for loading the plot result")
    parser.add_argument("--poisoned_ins_num", required=True, type=int,
                        help="the ground-truth poisoned instance number")
    parser.add_argument("--poisoned_feat_num", default=None, type=int,
                        help="the poisoned feature number. None means all poisoned")

    args = parser.parse_args()

    fig, ax = plt.subplots()
    ax.set_title("No Constrain on the Poisoned Feature Number" if args.poisoned_feat_num is None else
                 f"Poisoned Feature Number = {args.poisoned_feat_num}")
    ax.set_xlabel('Defender Assumed Poisoned Instance Number')
    ax.set_ylabel('Percentage of the Poisoned Test Dataset')
    x_max = 0
    folder = os.path.join(args.load_dir, args.load_folder)
    labels = ["Certified and Correct", "Certified But Wrong", "Abstain"]
    colors = ["limegreen", "red", "gray"]
    if os.path.exists(os.path.join(folder, f"plot_{args.poisoned_feat_num}.npy")):
        res = np.load(os.path.join(folder, f"plot_{args.poisoned_feat_num}.npy"), allow_pickle=True)
        x1 = [t for t in np.unique(res) if t >= 0]
        y1 = []
        p = int(np.sum(res < 0))
        for j in range(len(x1)):
            while res[p] != x1[j]: p += 1
            y1.append((len(res) - p) * 100.0 / len(res))
        x1.append(x1[-1] + 1)
        y1.append(0)
        if x1[0] != 0:
            y1 = [y1[0]] + y1
            x1 = [0] + x1

        x2 = [-(t + 2) for t in np.unique(res)[::-1] if t <= -2]
        y2 = []
        p = int(np.sum(res >= -1))
        for j in range(len(x2)):
            while res[len(res) - p - 1] != -x2[j] - 2: p += 1
            y2.append((len(res) - p) * 100.0 / len(res))
        if x2[0] != 0:
            y2 = [y2[0]] + y2
            x2 = [0] + x2

        p1 = 0
        p2 = 0
        y1_ = []
        y2_ = []
        x = []
        while p1 < len(y1) or p2 < len(y2):
            if p1 == len(y1) or (p2 < len(y2) and x1[p1] > x2[p2]):
                x.append(x2[p2])
                y2_.append(y2[p2])
                y1_.append(y1_[-1])
                p2 += 1
            elif p2 == len(y2) or (p1 < len(y1) and x1[p1] < x2[p2]):
                x.append(x1[p1])
                y1_.append(y1[p1])
                y2_.append(y2_[-1])
                p1 += 1
            else:
                assert x1[p1] == x2[p2]
                x.append(x1[p1])
                y1_.append(y1[p1])
                y2_.append(y2[p2])
                p1 += 1
                p2 += 1

        y3 = [100 - y1_[t] - y2_[t] for t in range(len(x))]

        ax.stackplot(x, y1_, y2_, y3, baseline="zero", colors=colors, labels=labels)
        ax.axvline(x=args.poisoned_ins_num, color='black', label='Actual Attacked Instance Number', linestyle='dashed',
                   linewidth=0.5)
        x_max = max(x_max, max(x))
    else:
        warnings.warn(f"{os.path.join(folder, f'plot_{args.poisoned_feat_num}.npy')} does not detected!")
        exit(0)

    ax.set(xlim=(0, x_max), ylim=(0, 100))
    ax.legend()
    # plt.show()
    plt.savefig(f"./{args.load_dir}/{args.save_file_name}.pdf")
