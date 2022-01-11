"""
From
"""
import numpy as np
from scipy.special import comb, factorial
from time import time
import os
from tqdm import tqdm
from fractions import Fraction

global_comb = dict()
global_powe = dict()


def my_comb(d, m):
    if (d, m) not in global_comb:
        global_comb[(d, m)] = comb(d, m, exact=True)

    return global_comb[(d, m)]


def my_powe(k, p):
    if (k, p) not in global_powe:
        global_powe[(k, p)] = k ** p

    return global_powe[(k, p)]


def process_count(Ia, Ib, fn, global_d, K, v):
    if not os.path.exists(f"list_counts/{fn}"):
        os.mkdir(f"list_counts/{fn}")

    if os.path.exists(f'list_counts/{fn}/complete_count_{v}_{float(Ia):.2f}_{float(Ib):.2f}.npz'):
        npzfile = np.load(f'list_counts/{fn}/complete_count_{v}_{float(Ia):.2f}_{float(Ib):.2f}.npz', allow_pickle=True)
        return npzfile["complete_cnt_p"], npzfile["complete_cnt_q"]

    m_range = [0, global_d + 1]  # m -> u in the paper, the number of feature flipped in x

    def get_count(d, m, n, v):
        if v == 0 and m == 0 and n == 0:
            return 1
        # early stopping
        if (v == 0 and m != n) or min(m, n) < 0 or max(m, n) > d or m + n < v:
            return 0

        if v == 0:
            return my_comb(d, m) * my_powe(K, m)
        else:
            c = 0
            # the number which are assigned to the (d-v) dimensions
            for i in range(max(0, n - v), min(m, d - v, int(np.floor((m + n - v) * 0.5))) + 1):
                if (m + n - v) / 2 < i:
                    break
                x = m - i
                y = n - i
                j = x + y - v
                # the second one implies n <= m+v
                if j < 0 or x < j:
                    continue
                tmp = my_powe(K - 1, j) * my_comb(v, x - j) * my_comb(v - x + j, j)
                if tmp != 0:
                    tmp *= my_comb(d - v, i) * my_powe(K, i)
                    c += tmp

            return c

    ttl = 0
    complete_cnt = []
    for m in range(m_range[0], m_range[1]):
        for n in range(m, min(m + v, global_d) + 1):  # n -> v the number of feature flipped in x'
            c = get_count(global_d, m, n, v)
            if c != 0:
                complete_cnt.append(((m, n), c))
                ttl += c  # * (n - m) * (a ** (global_d - m)) * (b ** m)
                # symmetric between d, m, n, v and d, n, m, v
                if n > m:
                    ttl += c  # * (m - n) * (a ** (global_d - n)) * (b ** n)

    complete_cnt_p = [0] * (global_d * 2 + 1)
    complete_cnt_q = [0] * (global_d * 2 + 1)
    for ((m, n), c) in tqdm(complete_cnt):
        complete_cnt_p[m - n + global_d] += c * (Ia ** (global_d - m)) * (Ib ** m)
        complete_cnt_q[m - n + global_d] += c * (Ia ** (global_d - n)) * (Ib ** n)
        if m != n:
            complete_cnt_p[n - m + global_d] += c * (Ia ** (global_d - n)) * (Ib ** n)
            complete_cnt_q[n - m + global_d] += c * (Ia ** (global_d - m)) * (Ib ** m)

    np.savez(f'list_counts/{fn}/complete_count_{v}_{float(Ia):.2f}_{float(Ib):.2f}',
             complete_cnt_p=complete_cnt_p,
             complete_cnt_q=complete_cnt_q)

    return complete_cnt_p, complete_cnt_q
