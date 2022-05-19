import os
import numpy as np
import argparse
import warnings
import csv

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--load_folder", type=str, help="folder for loading the npys", required=True)
    parser.add_argument("--load_files", type=str, help="npy files", required=True)
    parser.add_argument("--training_size", type=int, help="training set size", required=True)
    parser.add_argument("--point_num", type=int, help="point number in csv", default=100)
    parser.add_argument("--save_file", type=str, help="csv filename", default="temp")
    parser.add_argument("--normal_acc", action='store_true', help="whether to add normal accuracy")

    args = parser.parse_args()
    files = eval(args.load_files)  # list[str]

    x_max = 0
    X = []
    Y = []
    normal = []
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
            y = np.array(y)
            x = np.array(x) * 100.0 / args.training_size
            X.append(x)
            Y.append(y)
            normal.append(dash_y)
            x_max = max(x_max, x[-1])
        else:
            warnings.warn(f"{npy_file} does not detected!")

    with open(f'{os.path.join(args.load_folder, args.save_file)}.csv', 'w', newline='') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=',',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
        header = ['x'] + [str(_) for _ in range(len(files))]
        if args.normal_acc:
            header += [str(_) + "-normal" for _ in range(len(files))]
        spamwriter.writerow(header)
        p = [0] * len(files)
        for i in np.linspace(0, x_max, args.point_num):
            res = []
            for j in range(len(files)):
                while p[j] + 1 < len(X[j]) and X[j][p[j]] <= i: p[j] += 1
                if X[j][p[j]] < i:
                    res.append(0)
                else:
                    res.append(Y[j][p[j]])
            if args.normal_acc:
                for j in range(len(files)):
                    res.append(normal[j])

            spamwriter.writerow([i] + res)
