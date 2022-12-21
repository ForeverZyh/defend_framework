import torch
import os
import json
import time
import tqdm

import numpy as np

from utils.data_processing import EmberPoisonDataPreProcessor
from utils.cert_train_argments import get_arguments
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import NearestNeighbors

if __name__ == "__main__":
    parser = get_arguments()
    args = parser.parse_args()
    if args.exp_name is None:
        args.exp_name = time.strftime("%Y%m%d-%H%M%S")

    if args.model_save_dir is not None:
        if not os.path.exists(args.model_save_dir):
            os.mkdir(args.model_save_dir)

    res = None
    res_noise = None
    if args.res_save_dir is not None:
        if not os.path.exists(args.res_save_dir):
            os.mkdir(args.res_save_dir)
        if os.path.exists(os.path.join(args.res_save_dir, args.exp_name)):
            respond = input("Experiment already exists, type [O] to overwrite, type [R] to resume")
            if respond == "O":
                pass
            elif respond == "R":
                res, res_noise = np.load(os.path.join(args.res_save_dir, args.exp_name, "aggre_res.npy"))
            else:
                exit(0)
        else:
            os.mkdir(os.path.join(args.res_save_dir, args.exp_name))

    assert args.res_save_dir is not None and args.exp_name is not None
    data_loader = EmberPoisonDataPreProcessor(args)
    with open(os.path.join(args.res_save_dir, args.exp_name, "commandline_args.txt"), 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    minmax = MinMaxScaler(clip=True)
    data_loader.x_train = minmax.fit_transform(data_loader.x_train)
    data_loader.x_test = minmax.transform(data_loader.x_test)
    nbrs = NearestNeighbors(n_neighbors=args.N, algorithm='auto', p=1, n_jobs=16).fit(data_loader.x_train)
    nbrs.fit(data_loader.x_train)
    correct = np.array([])
    radius = np.array([])
    for i in tqdm.tqdm(range(0, len(data_loader.x_test), args.batch_size)):
        end = min(len(data_loader.x_test), i + args.batch_size)
        ret = nbrs.kneighbors(data_loader.x_test[i:end], return_distance=False)
        label_1 = np.sum(data_loader.y_train[ret], axis=-1)
        label_0 = args.N - label_1
        batch_correct = (label_1 > label_0) == data_loader.y_test[i:end]
        batch_radius = np.abs(label_1 - label_0) // 2
        correct = np.append(correct, batch_correct)
        radius = np.append(radius, batch_radius)
        print(f"Accuracy: {np.mean(correct)}\tCertified accuracy: {np.mean((radius > 600) * correct)}\tCertified wrong: {np.mean((radius > 600) * (1 - correct))}")

    np.save(os.path.join(args.res_save_dir, args.exp_name, "aggre_res"), (correct, radius))
