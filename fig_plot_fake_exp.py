import os
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import binom
from scipy.special import comb
from fractions import Fraction
from multiprocessing import Pool
import math

from cal_bound import FlipBoundCalculator, SelectBoundCalculator

dataset = "sst2"
fig, ax = plt.subplots()
ax.set_xlabel('r')
ax.set_ylabel('s')

if dataset == "mnist17":
    st_d = 0
    pa = 0.8
    k = 50
    D = 13007
    Rbag = int(np.ceil(D * (1 - np.power(1 - (pa - (1 - pa)) / 2, 1 / k))) - 1)
    print(Rbag)
    d = 28 * 28 + 1
    Rff = 3
    a = 0.8
    K = 1
    Ia = Fraction(8, 10)
    Ib = Fraction(2, 10 * K)
    bound_cal = FlipBoundCalculator(Ia, Ib, dataset, D, d, K, k)
elif dataset == "ember":
    pa = 0.8
    k = 500
    D = 600000
    Rbag = int(np.ceil(D * (1 - np.power(1 - (pa - (1 - pa)) / 2, 1 / k))) - 1)
    print(Rbag)
    d = 2351
    Rff = 3
    a = 0.7
    K = 1
    Ia = Fraction(7, 10)
    Ib = Fraction(3, 10 * K)
    bound_cal = FlipBoundCalculator(Ia, Ib, dataset, D, d, K, k)
elif dataset == "sst2":
    bound_cal = SelectBoundCalculator(None, dataset, 25000, 200, 2000, 100)
else:
    raise NotImplementedError


# print(len(ans))
# exit(0)


def f(args):
    m_n_delta, mn0, complete_cnt = args
    return complete_cnt[mn0]


print(bound_cal.get_pa_lb_binary(250, 5))
