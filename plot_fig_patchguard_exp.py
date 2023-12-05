import os
import matplotlib.pyplot as plt
import numpy as np
import argparse
import warnings
import csv


def draw(pecan, patchguard, name):
    # create a Pareto frontier
    fig, ax = plt.subplots()
    ax.set_title(f"{name} data")
    ax.set_ylabel('Correct')
    ax.set_xlabel('Wrong')
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_aspect('equal', adjustable='box')
    ax.grid(True)
    keys = sorted(list(pecan.keys()))
    xs = []
    ys = []
    for key in keys:
        xs.append(pecan[key][1])
        ys.append(pecan[key][0])
    # draw a line connecting the points without marker
    ax.plot(xs, ys, color="blue")
    # draw a point at (xs[-1], ys[-1])
    ax.scatter(xs[-1], ys[-1], color="blue", marker="x")
    ax.text(xs[-1] + 1, ys[-1] + 1, f"DPA only")
    keys = sorted(list(patchguard.keys()))
    xs = []
    ys = []
    for key in keys:
        xs.append(patchguard[key][1])
        ys.append(patchguard[key][0])
    ax.plot(xs, ys, color="red")
    ax.scatter(xs[-1], ys[-1], color="red", marker="x")
    ax.text(xs[-1] + 1, ys[-1] + 1, f"No Defense")
    # draw legend
    ax.legend(["PECAN", "DPA only", "PatchGuard++", "No Defense"])
    plt.savefig(os.path.join(args.load_dir_pecan, f"{name}.pdf"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--d", type=str, help="dataset", choices=["mnist", "cifar10-02"])
    parser.add_argument("--load_dir_pecan", type=str, help="folder for loading the pecan result")
    parser.add_argument("--load_dir_patchguard", type=str, help="folder for loading the patchguard result")
    args = parser.parse_args()
    args.clean_file = "plot_0.npy"
    args.poisoned_file = "plot_1.npy"
    args.patchguard_file = "0_predictions.npy"
    pecan_clean_dict = np.load(os.path.join(args.load_dir_pecan, args.clean_file), allow_pickle=True).item()
    pecan_poisoned_dict = np.load(os.path.join(args.load_dir_pecan, args.poisoned_file), allow_pickle=True).item()

    patchguard_res = np.load(os.path.join(args.load_dir_patchguard, args.patchguard_file), allow_pickle=True)
    patchguard_res_clean_pred, patchguard_res_clean_conf = patchguard_res[:2]
    patchguard_res_poisoned_pred, patchguard_res_poisoned_conf = patchguard_res[2:]
    res, _ = np.load(os.path.join(args.load_dir_pecan, "aggre_res.npy"))
    y = res[:, -1]
    y = y[:len(y) // 2]  # the labels of test data
    if args.d == "mnist":
        patchguard_res_poisoned_pred = patchguard_res_poisoned_pred[y != 0]  # the poisoned data
        patchguard_res_poisoned_conf = patchguard_res_poisoned_conf[y != 0]
        poisoned_y = y[y != 0]
    else:
        patchguard_res_poisoned_pred = patchguard_res_poisoned_pred[y == 1]  # the poisoned data
        patchguard_res_poisoned_conf = patchguard_res_poisoned_conf[y == 1]
        poisoned_y = y[y == 1]

    patchguard_clean_dict = {}
    patchguard_poisoned_dict = {}
    for label, pred, conf, d in [(y, patchguard_res_clean_pred, patchguard_res_clean_conf, patchguard_clean_dict),
                                 (poisoned_y, patchguard_res_poisoned_pred, patchguard_res_poisoned_conf,
                                  patchguard_poisoned_dict)]:
        for tau in np.linspace(0, 1, 101):
            verified = conf <= tau
            cert_correct = np.mean((pred == label) * verified) * 100
            cert_wrong = np.mean((pred != label) * verified) * 100
            abstain = 100 - cert_correct - cert_wrong
            d[tau] = (cert_correct, cert_wrong, abstain)
            # print(tau, cert_correct, cert_wrong, abstain)

    draw(pecan_poisoned_dict, patchguard_poisoned_dict, "poisoned")
    draw(pecan_clean_dict, patchguard_clean_dict, "clean")
