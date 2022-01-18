import os
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import binom
from scipy.special import comb
from fractions import Fraction
from multiprocessing import Pool
import math

from cal_bound import BoundCalculator

dataset = "mnist17"
considered_degree = 2
algorithm = "NP"
assert algorithm in ["NP", "NP+KL"]
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
elif dataset == "ember":
    # st_d = 600
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
bound_cal = BoundCalculator(Ia, Ib, dataset, D, d, K, k, considered_degree=considered_degree, algorithm=algorithm)

ax.plot([Rbag, Rbag], [1, d], label="bagging")
ax.plot(np.arange(D + 1),
        [d if x == 0 else min(d, np.floor_divide(Rff, x)) for x in range(D + 1)], label="feature-flipping")

filepath_prefix = "../Randomized_Smoothing/MNIST_exp/compute_rho/list_counts/%s/" % dataset

"""
considered_degree = 1
ans = [785, 785, 785, 785, 785, 785, 785, 785, 785, 785, 785, 785, 785, 785, 785, 785, 785, 785, 785, 785, 785, 785,
       785, 785, 785, 785, 785, 785, 785, 785, 785, 785, 785, 785, 785, 785, 785, 785, 785, 785, 785, 785, 785, 785,
       785, 785, 785, 785, 785, 785, 785, 785, 785, 785, 785, 785, 785, 785, 785, 785, 785, 785, 785, 785, 785, 785,
       785, 785, 785, 785, 785, 785, 785, 785, 785, 785, 785, 785, 785, 785, 785, 785, 785, 785, 785, 785, 785, 785,
       785, 785, 785, 785, 785, 44, 32, 28, 24, 22, 20, 18, 18, 16, 16, 14, 14, 14, 12, 12, 12, 10, 10, 10, 10, 10, 10,
       8, 8, 8, 8, 8, 8, 8, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 2, 2,
       2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]

considered_degree = 2
ans = [785, 785, 785, 785, 785, 785, 785, 785, 785, 785, 785, 785, 785, 785, 785, 785, 785, 785, 785, 785, 785, 785,
       785, 785, 785, 785, 785, 785, 785, 785, 785, 785, 785, 785, 785, 785, 785, 785, 785, 785, 785, 785, 785, 785,
       785, 785, 785, 785, 785, 785, 785, 785, 785, 785, 785, 785, 785, 785, 785, 785, 785, 785, 785, 785, 785, 785,
       785, 785, 785, 785, 785, 785, 785, 785, 785, 785, 785, 785, 785, 785, 785, 785, 785, 785, 785, 785, 785, 785,
       785, 785, 785, 785, 785, 44, 32, 28, 24, 22, 20, 19, 18, 16, 16, 15, 14, 14, 13, 12, 12, 12, 12, 10, 10, 10, 10,
       10, 9, 8, 8, 8, 8, 8, 8, 8, 8, 7, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 5, 5, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
       4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 3, 3, 3, 3, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
       2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
       2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
       
print(len(ans)) 186 vs. 259

considered_degree = 1 (KL_bounds)
ans = [785, 785, 785, 785, 785, 785, 785, 785, 785, 785, 785, 785, 785, 785, 785, 785, 785, 785, 785, 785, 785, 785,
       785, 785, 785, 785, 785, 785, 785, 785, 785, 785, 785, 785, 785, 785, 785, 785, 785, 785, 785, 785, 785, 785,
       785, 785, 785, 785, 785, 785, 785, 785, 785, 785, 785, 785, 785, 785, 785, 785, 785, 785, 785, 785, 785, 785,
       785, 785, 785, 785, 785, 785, 785, 785, 785, 785, 785, 785, 785, 785, 785, 785, 785, 785, 785, 785, 785, 785,
       785, 785, 785, 785, 785, 44, 32, 28, 24, 22, 20, 18, 18, 16, 16, 14, 14, 14, 12, 12, 12, 12, 10, 10, 10, 10, 10,
       8, 8, 8, 8, 8, 8, 8, 8, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 5, 5, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
       4, 4, 4, 4, 4, 4, 3, 3, 3, 3, 3, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
       2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1,
       1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

considered_degree = 2 (KL_bounds)
ans = [785, 785, 785, 785, 785, 785, 785, 785, 785, 785, 785, 785, 785, 785, 785, 785, 785, 785, 785, 785, 785, 785,
       785, 785, 785, 785, 785, 785, 785, 785, 785, 785, 785, 785, 785, 785, 785, 785, 785, 785, 785, 785, 785, 785,
       785, 785, 785, 785, 785, 785, 785, 785, 785, 785, 785, 785, 785, 785, 785, 785, 785, 785, 785, 785, 785, 785,
       785, 785, 785, 785, 785, 785, 785, 785, 785, 785, 785, 785, 785, 785, 785, 785, 785, 785, 785, 785, 785, 785,
       785, 785, 785, 785, 785, 44, 32, 28, 24, 22, 20, 19, 18, 16, 16, 15, 14, 14, 13, 12, 12, 12, 12, 10, 10, 10, 10,
       10, 9, 8, 8, 8, 8, 8, 8, 8, 8, 7, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 5, 5, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
       4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 3, 3, 3, 3, 3, 3, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
       2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
       2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
       1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]     
       
print(len(ans)) 265 vs. 293
"""


# print(len(ans))
# exit(0)


def f(args):
    m_n_delta, mn0, complete_cnt = args
    return complete_cnt[mn0]


print(bound_cal.get_pa_lb_binary(184, 8))

# print(check_radius(600, 600, 4, 0.8))
exit(0)
ans = []
for x in range(D + 1):
    if x <= Rbag:
        ans.append(d)
        continue
    p_binom = [None] * (k + 1)
    other_pk = Fraction(1)
    for i in range(considered_degree + 1):
        p_binom[i] = comb(k, i, exact=True) * (Fraction(x, D) ** i) * ((1 - Fraction(x, D)) ** (k - i))
        other_pk -= p_binom[i]

    remain_to_assign = Fraction(pa) - other_pk
    if remain_to_assign <= 0:
        ans.append(0)
        continue
    # print(x, remain_to_assign)

    feasible_l = -1
    for l in range(20):
        if algorithm == "NP":
            if ans[-1] - 2 ** l + 1 <= 0 or bound_cal.check_NP_binary(ans[-1] - 2 ** l + 1, Fraction(pa) - other_pk,
                                                                      p_binom):
                feasible_l = l
                break
        elif algorithm == "NP+KL":
            if ans[-1] - 2 ** l + 1 <= 0 or bound_cal.check_NP_KL_binary(ans[-1] - 2 ** l + 1, x, Fraction(pa),
                                                                         other_pk, p_binom):
                feasible_l = l
                break
        else:
            raise NotImplementedError

    print(f"feasible_l: {feasible_l}")
    ans.append(max(0, ans[-1] - 2 ** feasible_l + 1))
    for l in range(feasible_l - 1, -1, -1):
        if 2 ** l + ans[-1] > ans[-2]:
            continue
        if algorithm == "NP":
            if bound_cal.check_NP_binary(ans[-1] + 2 ** l, Fraction(pa) - other_pk, p_binom):
                ans[-1] += 2 ** l
        elif algorithm == "NP+KL":
            if bound_cal.check_NP_KL_binary(ans[-1] + 2 ** l, x, Fraction(pa), other_pk, p_binom):
                ans[-1] += 2 ** l
        else:
            raise NotImplementedError

    print(x, ans[-1])

print(ans)
# exit(0)
# ans = [d if x == 0 else min(d, np.ceil(np.log(4 * pa * (1 - pa)) * D / 2 / k / x / np.log((1 - a) / a) / 0.4) - 1) for x
#        in range(D + 1)]
for x in range(D + 1):
    if len(ans) <= x:
        ans.append(0)
for x in range(len(ans)):
    if ans[x] != d:
        print(x - 1)
        break
print(ans[-1])
ax.plot(np.arange(D + 1), ans, label="our approach")

ax.set(xlim=(0, 200 + 1),
       ylim=(0, d + 1))
ax.legend()
plt.savefig("./%dNPL_bounds.pdf" % considered_degree)
plt.show()
