import os
import matplotlib.pyplot as plt
import numpy as np
import argparse
import warnings
import csv


def draw(pecan, patchguard, friendly_noise_acc, name, best_id=None, best_id1=None, fpa_acc=None):
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
    if best_id is None:
        print(np.array(ys) - np.array(xs))
        best_id = np.argmax(np.array(ys) - np.array(xs))
    # draw a line connecting the points without marker
    line1, = ax.plot(xs, ys, color="blue", linestyle=":", label="_")
    # draw a point at (xs[-1], ys[-1])
    ax.scatter(xs[best_id], ys[best_id], color="blue", marker="x", label="PECAN")
    print(best_id)
    print(xs[best_id], ys[best_id], f"PECAN")
    ax.scatter(xs[-1], ys[-1], color="blue", marker="o", label="DPA only")
    print(xs[-1], ys[-1], f"DPA only")
    keys = sorted(list(patchguard.keys()))
    xs = []
    ys = []
    for key in keys:
        xs.append(patchguard[key][0][1])
        ys.append(patchguard[key][0][0])
    if best_id1 is None:
        best_id1 = np.argmax(np.array(ys) - np.array(xs))
    line2, = ax.plot(xs, ys, color="red", linestyle=":", label="_")
    ax.scatter(xs[best_id1], ys[best_id1], color="red", marker="^", label="PatchGuard++")
    print(best_id1)
    print(xs[best_id1], ys[best_id1], f"PatchGuard++")
    print(patchguard[keys[best_id1]][1][1], patchguard[keys[best_id1]][1][0], patchguard[keys[best_id1]][1][2],
          f"PatchGuard++ Variance")
    ax.scatter(xs[-1], ys[-1], color="red", marker="v", label="No Defense")
    print(xs[-1], ys[-1], f"No Defense")
    print(patchguard[keys[-1]][1][1], patchguard[keys[-1]][1][0], patchguard[keys[-1]][1][2], f"No Defense Variance")
    print(100 - friendly_noise_acc, friendly_noise_acc, f"Friendly Noise")
    ax.scatter(100 - friendly_noise_acc, friendly_noise_acc, color="green", marker="<", label="Friendly Noise")
    if fpa_acc is not None:
        ax.scatter(fpa_acc[1], fpa_acc[0], color="orange", marker=">", label="FPA")
        print(fpa_acc[1], fpa_acc[0], f"FPA")
    # ax.text(100 - friendly_noise_acc + 1, friendly_noise_acc + 1, f"Friendly Noise")
    # draw legend
    ax.legend()
    plt.savefig(os.path.join(args.load_dir_pecan, f"{name}.pdf"))
    return best_id, best_id1


def get_patchguard_dicts(patchguard_file, indices):
    patchguard_res = np.load(os.path.join(args.load_dir_patchguard, patchguard_file), allow_pickle=True)
    patchguard_res_clean_pred, patchguard_res_clean_conf = patchguard_res[:2]
    patchguard_res_poisoned_pred, patchguard_res_poisoned_conf = patchguard_res[2:]

    patchguard_res_poisoned_pred = patchguard_res_poisoned_pred[indices]  # the poisoned data
    patchguard_res_poisoned_conf = patchguard_res_poisoned_conf[indices]
    poisoned_y = y[indices]

    patchguard_res_poisoned_pred_ctrl_indices = (patchguard_res_clean_pred[indices] == poisoned_y) & (
            patchguard_res_clean_conf[indices] <= 1)

    patchguard_clean_dict = {}
    patchguard_poisoned_dict = {}
    patchguard_poisoned_dict_ctrl = {}
    for label, pred, conf, d in [(y, patchguard_res_clean_pred, patchguard_res_clean_conf, patchguard_clean_dict),
                                 (poisoned_y, patchguard_res_poisoned_pred, patchguard_res_poisoned_conf,
                                  patchguard_poisoned_dict)]:
        for tau in np.linspace(0, 1, 101):
            verified = conf <= tau
            cert_correct = np.mean((pred == label) * verified) * 100
            cert_wrong = np.mean((pred != label) * verified) * 100
            abstain = 100 - cert_correct - cert_wrong
            d[tau] = np.array([cert_correct, cert_wrong, abstain])
            # print(tau, cert_correct, cert_wrong, abstain)

    for tau in np.linspace(0, 1, 101):
        label = poisoned_y[patchguard_res_poisoned_pred_ctrl_indices]
        pred = patchguard_res_poisoned_pred[patchguard_res_poisoned_pred_ctrl_indices]
        conf = patchguard_res_poisoned_conf[patchguard_res_poisoned_pred_ctrl_indices]
        verified = conf <= tau
        cert_correct = np.mean((pred == label) * verified) * 100
        cert_wrong = np.mean((pred != label) * verified) * 100
        abstain = 100 - cert_correct - cert_wrong
        patchguard_poisoned_dict_ctrl[tau] = np.array([cert_correct, cert_wrong, abstain])
        # print((cert_correct, cert_wrong, abstain))

    return patchguard_clean_dict, patchguard_poisoned_dict, patchguard_poisoned_dict_ctrl


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--d", type=str, help="dataset", choices=["mnist", "cifar10-02"])
    parser.add_argument("--load_dir_pecan", type=str, help="folder for loading the pecan result")
    parser.add_argument("--load_dir_patchguard", type=str, help="folder for loading the patchguard result")
    args = parser.parse_args()
    args.clean_file = "plot_0.npy"
    args.poisoned_file = "plot_1.npy"
    pecan_clean_dict_ = np.load(os.path.join(args.load_dir_pecan, args.clean_file), allow_pickle=True).item()
    pecan_clean_dict = dict([(k, v[0]) for k, v in pecan_clean_dict_.items()])
    pecan_poisoned_dict_ = np.load(os.path.join(args.load_dir_pecan, args.poisoned_file), allow_pickle=True).item()
    pecan_poisoned_dict = dict([(k, v[0]) for k, v in pecan_poisoned_dict_.items()])
    pecan_poisoned_dict_ctrl = dict([(k, v[1]) for k, v in pecan_poisoned_dict_.items()])

    res, _ = np.load(os.path.join(args.load_dir_pecan, "aggre_res.npy"))
    y = res[:, -1]
    y = y[:len(y) // 2]  # the labels of test data
    if args.d == "mnist":
        indices = y != 0
    else:
        indices = y == 1

    patchguard_clean_dict = {}
    patchguard_poisoned_dict = {}
    patchguard_poisoned_dict_ctrl = {}
    for repeat_num in range(5):
        patchguard_file = f"{repeat_num}_predictions.npy"
        c, p, pc = get_patchguard_dicts(patchguard_file, indices)
        for a, b in zip([patchguard_clean_dict, patchguard_poisoned_dict, patchguard_poisoned_dict_ctrl], [c, p, pc]):
            for k, v in b.items():
                if k not in a:
                    a[k] = []
                a[k].append(v)

    for a in [patchguard_clean_dict, patchguard_poisoned_dict, patchguard_poisoned_dict_ctrl]:
        for k, v in a.items():
            a[k] = (np.mean(np.array(v), axis=0), np.std(np.array(v), axis=0))

    # (accuracy on poisoned data, accuracy on clean data, accuracy on poisoned data with control)
    if args.d == "mnist":
        a = [71.24, 73.82, 75.49, 85.67, 71.39]
        b = [79.51, 84.78, 88.66, 90.97, 82.74]
        c = [90.58, 87.02, 84.99, 93.8, 86.58]
        friendlynoise = (np.mean(a),
                         np.mean(b),
                         np.mean(c))
        friendlynoise_var = (np.std(a),
                             np.std(b),
                             np.std(c))
        fpa = ([0, 0], [0, 0], [0, 0])
    else:
        a = [69, 74.9, 72.9, 69.3, 72.5]
        b = [85.9, 86.25, 86.05, 86.3, 85.6]
        c = [80.14, 87.37, 84.67, 79.56, 84.58]
        friendlynoise = (np.mean(a),
                         np.mean(b),
                         np.mean(c))
        friendlynoise_var = (np.std(a),
                             np.std(b),
                             np.std(c))
        fpa = ([0, 0], [0, 0], [0, 0])

    best_id, best_id1 = draw(pecan_poisoned_dict_ctrl, patchguard_poisoned_dict_ctrl, friendlynoise[2], "control",
                             fpa_acc=fpa[2])
    draw(pecan_poisoned_dict, patchguard_poisoned_dict, friendlynoise[0], "poisoned", best_id=best_id,
         best_id1=best_id1, fpa_acc=fpa[0])
    draw(pecan_clean_dict, patchguard_clean_dict, friendlynoise[1], "clean", best_id=best_id, best_id1=best_id1,
         fpa_acc=fpa[1])
    print(friendlynoise_var)
