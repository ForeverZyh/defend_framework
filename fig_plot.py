import os
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import binom
from scipy.special import comb
from fractions import Fraction
from tqdm import trange, tqdm
from multiprocessing import Pool
import math

dataset = "ember"
algorithm = "NP+KL"
assert algorithm in ["NP", "NP+KL"]
fig, ax = plt.subplots()
ax.set_xlabel('r')
ax.set_ylabel('s')

if dataset == "mnist":
    st_d = 0
    pa = 0.8
    k = 50
    D = 13007
    Rbag = int(np.ceil(D * (1 - np.power(1 - (pa - (1 - pa)) / 2, 1 / k))) - 1)
    print(Rbag)
    d = 28 * 28 + 1
    Rff = 3
    a = 0.7
    K = 1
    Ia = Fraction(7, 10)
    Ib = Fraction(3, 10 * K)
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

ax.plot([Rbag, Rbag], [1, d], label="bagging")
ax.plot(np.arange(D + 1),
        [d if x == 0 else min(d, np.floor_divide(Rff, x)) for x in range(D + 1)], label="feature-flipping")

filepath_prefix = "../Randomized_Smoothing/MNIST_exp/compute_rho/list_counts/%s/" % dataset
considered_degree = 2

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


def log(a):
    return Fraction(math.log(a.numerator)) - Fraction(math.log(a.denominator))


def get_KL_divergence_cf(theta_a, kl):
    theta_a = Fraction(theta_a)
    if theta_a == 1:
        return 1
    l = 0
    r = theta_a
    theta_b = 1 - theta_a
    for i in range(50):
        cf = (l + r) / 2
        if kl <= (1 - cf) * (log(1 - cf) - log(theta_b)) + cf * log(cf) - cf * log(theta_a):
            l = cf
        else:
            r = cf

    return l


# print(float(
#     get_KL_divergence_cf(1 - Fraction(1, 1000000000000000000),
#                          ((k - 2) * 0.01 + 2) * np.log(a / (1 - a)) * 8 * (a - 1 + a))))
# exit(0)


# actually runs slower for considered_degree = 1, but much faster when considered_degree = 2.
# the Fraction in the preprocessing is to blame.
def check_NP(s, remain_to_assign):
    try:
        complete_cnt_ = list(np.load(os.path.join(filepath_prefix, 'complete_count_{}_0.npy'.format(s)),
                                     allow_pickle=True))
    except:
        return False
    achieved = 0

    complete_cnt_p = [0] * (d * 2 + 1)
    complete_cnt_q = [0] * (d * 2 + 1)
    for ((m, n), c) in tqdm(complete_cnt_):
        complete_cnt_p[m - n + d] += c * (Ia ** (d - m)) * (Ib ** m)
        complete_cnt_q[m - n + d] += c * (Ia ** (d - n)) * (Ib ** n)
        if m != n:
            complete_cnt_p[n - m + d] += c * (Ia ** (d - n)) * (Ib ** n)
            complete_cnt_q[n - m + d] += c * (Ia ** (d - m)) * (Ib ** m)

    for m_n_delta in trange(-considered_degree * d, considered_degree * d + 1):
        outcome = []
        # pos_cnt = 0
        if m_n_delta == 0:
            outcome.append([1, 1, 0])
        # pos_cnt = 1
        if considered_degree >= 1 and d >= abs(m_n_delta) and complete_cnt_p[m_n_delta + d] > 0:
            outcome.append([complete_cnt_p[m_n_delta + d], complete_cnt_q[m_n_delta + d], 1])
        # pos_cnt = 2
        if considered_degree >= 2:
            cnt_p = sum(complete_cnt_p[mn0 + d] * complete_cnt_p[m_n_delta - mn0 + d] for mn0 in
                        range(max(-d, m_n_delta - d), min(d, m_n_delta + d) + 1))
            cnt_q = sum(complete_cnt_q[mn0 + d] * complete_cnt_q[m_n_delta - mn0 + d] for mn0 in
                        range(max(-d, m_n_delta - d), min(d, m_n_delta + d) + 1))
            if cnt_p > 0:
                outcome.append([cnt_p, cnt_q, 2])

        for i in range(len(outcome)):
            p_cnt, q_cnt, poison_cnt = outcome[i]

            q_delta = q_cnt * p_binom[poison_cnt]
            p_delta = p_cnt * p_binom[poison_cnt]

            if p_delta < remain_to_assign:
                remain_to_assign -= p_delta
                achieved += q_delta
                if achieved > Fraction(1, 2):
                    del complete_cnt_p
                    del complete_cnt_q
                    del complete_cnt_
                    return True
            else:
                achieved += remain_to_assign / ((Ia ** (-m_n_delta)) * (Ib ** m_n_delta))
                del complete_cnt_p
                del complete_cnt_q
                del complete_cnt_
                return achieved > Fraction(1, 2)

    del complete_cnt_p
    del complete_cnt_q
    del complete_cnt_
    return achieved > Fraction(1, 2)


def check_NP_KL(s, r, p, _1mp1):
    p1 = 1 - _1mp1
    try:
        complete_cnt_ = list(np.load(os.path.join(filepath_prefix, 'complete_count_{}_0.npy'.format(s)),
                                     allow_pickle=True))
    except:
        return False
    achieved = 0
    assigned = 0

    complete_cnt_p = [0] * (d * 2 + 1)
    complete_cnt_q = [0] * (d * 2 + 1)
    for ((m, n), c) in tqdm(complete_cnt_):
        complete_cnt_p[m - n + d] += c * (Ia ** (d - m)) * (Ib ** m)
        complete_cnt_q[m - n + d] += c * (Ia ** (d - n)) * (Ib ** n)
        if m != n:
            complete_cnt_p[n - m + d] += c * (Ia ** (d - n)) * (Ib ** n)
            complete_cnt_q[n - m + d] += c * (Ia ** (d - m)) * (Ib ** m)

    for m_n_delta in trange(-considered_degree * d, considered_degree * d + 1):
        outcome = []
        # pos_cnt = 0
        if m_n_delta == 0:
            outcome.append([1, 1, 0])
        # pos_cnt = 1
        if considered_degree >= 1 and d >= abs(m_n_delta) and complete_cnt_p[m_n_delta + d] > 0:
            outcome.append([complete_cnt_p[m_n_delta + d], complete_cnt_q[m_n_delta + d], 1])
        # pos_cnt = 2
        if considered_degree >= 2:
            cnt_p = sum(complete_cnt_p[mn0 + d] * complete_cnt_p[m_n_delta - mn0 + d] for mn0 in
                        range(max(-d, m_n_delta - d), min(d, m_n_delta + d) + 1))
            cnt_q = sum(complete_cnt_q[mn0 + d] * complete_cnt_q[m_n_delta - mn0 + d] for mn0 in
                        range(max(-d, m_n_delta - d), min(d, m_n_delta + d) + 1))
            if cnt_p > 0:
                outcome.append([cnt_p, cnt_q, 2])

        for i in range(len(outcome)):
            p_cnt, q_cnt, poison_cnt = outcome[i]
            ratio = ((Ia ** (-m_n_delta)) * (Ib ** m_n_delta))

            q_delta = q_cnt * p_binom[poison_cnt]
            p_delta = p_cnt * p_binom[poison_cnt]

            start = True
            if assigned < p1 + p - 1 < assigned + p_delta:
                start_l = p1 + p - 1
            elif assigned >= p1 + p - 1:
                start_l = assigned
            else:
                start = False

            end = False
            if assigned + p_delta > min(p1, p):
                start_r = min(p1, p)
                end = True
            else:
                start_r = assigned + p_delta

            if start:
                # what if in the middle?
                # tenary search
                value_mid_l = None
                value_mid_r = None

                def get_value(mid):
                    return achieved + (mid - assigned) / ratio + _1mp1 * get_KL_divergence_cf(
                        (p - mid) / _1mp1,
                        ((k - considered_degree) * Fraction(r, D) + considered_degree) * log(Ia / Ib) * s * (Ia - Ib))

                for _ in range(100):
                    mid_l = (start_l * 2 + start_r) / 3
                    mid_r = (start_l + start_r * 2) / 3
                    value_mid_l = get_value(mid_l)
                    value_mid_r = get_value(mid_r)

                    if value_mid_l > value_mid_r:
                        start_l = mid_l
                    else:
                        start_r = mid_r

                    if value_mid_l <= Fraction(1, 2) or value_mid_r <= Fraction(1, 2):
                        return False

            if end:
                break

            assigned += p_delta
            achieved += q_delta

            if achieved > Fraction(1, 2):
                del complete_cnt_p
                del complete_cnt_q
                del complete_cnt_
                return True

    return True


def check_radius(x, k, s, pa):
    """
    return whether the radius is certifiable
    :param x: the number of poisoned instance
    :param k: the size of each bag
    :param s: the number of poisoned feature
    :return: bool, whether it is certifiable
    """
    global p_binom
    p_binom = [None] * (k + 1)
    other_pk = Fraction(1)
    for i in range(considered_degree + 1):
        p_binom[i] = comb(k, i, exact=True) * (Fraction(x, D) ** i) * ((1 - Fraction(x, D)) ** (k - i))
        other_pk -= p_binom[i]

    if algorithm == "NP":
        return check_NP(s, Fraction(pa) - other_pk)
    elif algorithm == "NP+KL":
        return check_NP_KL(s, x, Fraction(pa), other_pk)
    else:
        raise NotImplementedError


l_pa = Fraction(0)
r_pa = Fraction(1)
for i in range(50):
    mid = (l_pa + r_pa) / 2
    if check_radius(600, 500, 4, mid):
        r_pa = mid
    else:
        l_pa = mid
print(r_pa)

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
            if ans[-1] - 2 ** l + 1 <= 0 or check_NP(ans[-1] - 2 ** l + 1, Fraction(pa) - other_pk):
                feasible_l = l
                break
        elif algorithm == "NP+KL":
            if ans[-1] - 2 ** l + 1 <= 0 or check_NP_KL(ans[-1] - 2 ** l + 1, x, Fraction(pa), other_pk):
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
            if check_NP(ans[-1] + 2 ** l, Fraction(pa) - other_pk):
                ans[-1] += 2 ** l
        elif algorithm == "NP+KL":
            if check_NP_KL(ans[-1] + 2 ** l, x, Fraction(pa), other_pk):
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
