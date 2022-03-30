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
    labels = ["Certfied and Corret", "Certified But Wrong", "Abstin"]
    colors = ["limegreen", "red", "gray"]
    if os.path.exists(os.path.join(folder, f"plot_{args.poisoned_feat_num}.npy")):
        res = dict(np.load(os.path.join(folder, f"plot_{args.poisoned_feat_num}.npy"), allow_pickle=True).item())
        x = sorted(list(res.keys()))
        y1 = [np.mean(res[i] == 1) * 100 for i in x]
        y2 = [np.mean(res[i] == -1) * 100 for i in x]
        y3 = [np.mean(res[i] == 0) * 100 for i in x]
        ax.stackplot(x, y1, y2, y3, baseline="zero", colors=colors, labels=labels)
        ax.axvline(x=600, color='black', label='Actual Attacked Instance Number', linestyle='dashed', linewidth=0.5)
        x_max = max(x_max, max(x))
    else:
        warnings.warn(f"{os.path.join(folder, f'plot_{args.poisoned_feat_num}.npy')} does not detected!")
        exit(0)

    ax.set(xlim=(0, 1000), ylim=(0, 100))
    ax.legend()
    plt.show()
    plt.savefig(f"./{args.load_dir}/{args.save_file_name}.pdf")
