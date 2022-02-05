from fractions import Fraction
import math
import os
import numpy as np
from tqdm import trange, tqdm
from scipy.special import comb

from preprocessing_counts import process_count


def log(a):
    return Fraction(math.log(a.numerator)) - Fraction(math.log(a.denominator))


def get_KL_divergence_cf(theta_a, kl, theta_b=None):
    theta_a = Fraction(theta_a)
    if theta_a == 1:
        return 1
    l = 0
    r = theta_a
    if theta_b is None:
        theta_b = 1 - theta_a
    for i in range(50):
        cf = (l + r) / 2
        if kl <= (1 - cf) * (log(1 - cf) - log(theta_b)) + cf * log(cf) - cf * log(theta_a):
            l = cf
        else:
            r = cf

    return l


class BoundCalculator:
    def __init__(self, Ia, Ib, dataset, D, d, K, k):
        self.Ia = Ia
        self.Ib = Ib
        self.K = K
        self.k = k

        self.fn = dataset
        if not os.path.exists(os.path.join("list_counts", dataset)):
            os.mkdir(os.path.join("list_counts", dataset))

        self.D = D
        self.d = d
        self.cache_file = os.path.join("list_counts", dataset,
                                       f"cache_{str(Ia).replace('/', '__')}_{str(Ib).replace('/', '__')}_{K}_{k}_{d}")
        try:
            self.pa_lb_cache = np.load(self.cache_file + ".npy", allow_pickle=True).item()
        except:
            self.pa_lb_cache = dict()

    def check_NP_binary(self, s, pa, p_binom):
        return self.cal_NP_bound(s, pa, p_binom, early_stop=Fraction(1, 2)) > Fraction(1, 2)

    def check_NP(self, s, pa, pb, p_binom):
        """
        :param s: the number of poisoned features
        :param pa:
        :param pb:
        :param p_binom:
        :return: whether it is certifiable
        """
        lb = min(self.cal_NP_bound(s, pb, p_binom, reverse=True, early_stop=Fraction(1, 2)), Fraction(1, 2))
        return self.cal_NP_bound(s, pa, p_binom, early_stop=lb) > lb

    def cal_NP_bound(self, s, remain_to_assign, p_binom, reverse=False, early_stop=None):
        """
        return the lower bound (or upper bound if reverse is True) of classification result being y* in the
        original distribution
        :param s: the number of poisoned features
        :param remain_to_assign: the p of classification result being y* in the poisoned distribution
        :param p_binom: the pmf of the binomial distribution
        :param reverse: return upper bound (True) or lower bound (False)
        :param early_stop: if the return value is greater or equal to the early_stop value,
        we just return the current value
        :return: the lower bound (or upper bound if reverse is True)
        """
        Ia, Ib, fn, D, d, K, k = self.Ia, self.Ib, self.fn, self.D, self.d, self.K, self.k
        achieved = 0

        complete_cnt_p, complete_cnt_q = process_count(Ia, Ib, d, K, s)

        run_name = f'conv_count_{s}_{K}_{str(Ia).replace("/", "__")}_{str(Ib).replace("/", "__")}_{d}_{k}'
        filename = f'list_counts/{run_name}.npz'
        if os.path.exists(filename):
            npzfile = np.load(filename, allow_pickle=True)
            complete_cnt_ps, complete_cnt_qs = npzfile["complete_cnt_ps"], npzfile["complete_cnt_qs"]
        else:
            print("preparing " + run_name)
            complete_cnt_ps = [[1], complete_cnt_p]
            complete_cnt_qs = [[1], complete_cnt_q]
            # compute convolutions
            for i in range(2, self.k + 1):
                complete_cnt_ps.append([0] * (i * d * 2 + 1))
                complete_cnt_qs.append([0] * (i * d * 2 + 1))
                for j in range(d * 2 + 1):
                    for k in range(d * (i - 1) * 2 + 1):
                        complete_cnt_ps[i][j + k] += complete_cnt_ps[1][j] * complete_cnt_ps[i - 1][k]
                        complete_cnt_qs[i][j + k] += complete_cnt_qs[1][j] * complete_cnt_qs[i - 1][k]
            np.savez(filename,
                     complete_cnt_ps=complete_cnt_ps,
                     complete_cnt_qs=complete_cnt_qs)
            print("save file " + run_name + ".npz")

        if not reverse:
            _range = range(-self.k * d, self.k * d + 1)
        else:
            _range = range(self.k * d, -self.k * d - 1, -1)

        for m_n_delta in _range:
            outcome = []
            for i in range(self.k + 1):
                if -i * d <= m_n_delta <= i * d:
                    outcome.append([complete_cnt_ps[i][m_n_delta + i * d], complete_cnt_qs[i][m_n_delta + i * d], i])

            for i in range(len(outcome)):
                p_cnt, q_cnt, poison_cnt = outcome[i]

                q_delta = q_cnt * p_binom[poison_cnt]
                p_delta = p_cnt * p_binom[poison_cnt]

                if p_delta < remain_to_assign:
                    remain_to_assign -= p_delta
                    achieved += q_delta
                    if early_stop is not None and achieved > early_stop:
                        return achieved
                else:
                    achieved += remain_to_assign / ((Ia ** (-m_n_delta)) * (Ib ** m_n_delta))
                    return achieved

        return achieved

    def check_radius_binary(self, x, s, pa):
        """
        return whether the radius is certifiable
        :param x: the number of poisoned instance
        :param s: the number of poisoned feature
        :param pa: the lower bound of the probability of the most likely label
        :return: bool, whether it is certifiable
        """
        p_binom = [None] * (self.k + 1)
        for i in range(self.k + 1):
            p_binom[i] = comb(self.k, i, exact=True) * (Fraction(x, self.D) ** i) * (
                    (1 - Fraction(x, self.D)) ** (self.k - i))

        return self.check_NP_binary(s, Fraction(pa), p_binom)

    def check_radius(self, x, s, pa, pb):
        """
        return whether the radius is certifiable
        :param x: the number of poisoned instance
        :param s: the number of poisoned feature
        :param pa: the lower bound of the probability of the most likely label
        :param pb: the upper bound of the probability of the second most likely label
        :return: bool, whether it is certifiable
        """
        p_binom = [None] * (self.k + 1)
        for i in range(self.k + 1):
            p_binom[i] = comb(self.k, i, exact=True) * (Fraction(x, self.D) ** i) * (
                    (1 - Fraction(x, self.D)) ** (self.k - i))

        return self.check_NP(s, Fraction(pa), Fraction(pb), p_binom)

    def get_poisoned_ins_ub_binary(self, poisoned_feat_num, p_a):
        # binary search O(2log(ans))
        if not self.check_radius_binary(0, poisoned_feat_num, p_a):
            return -1
        else:
            feasible_l = -1
            for l in range(20):
                if not self.check_radius_binary(2 ** l, poisoned_feat_num, p_a):
                    feasible_l = l - 1
                    break

            if feasible_l == -1:
                return 0
            else:
                ans = 2 ** feasible_l
                for l in range(feasible_l - 1, -1, -1):
                    if self.check_radius_binary(ans + 2 ** l, poisoned_feat_num, p_a):
                        ans += 2 ** l
                return ans

    def get_pa_lb_binary(self, poisoned_ins_num, poisoned_feat_num):
        if (poisoned_ins_num, poisoned_feat_num) in self.pa_lb_cache:
            return self.pa_lb_cache[(poisoned_ins_num, poisoned_feat_num)]
        l_pa = Fraction(0)
        r_pa = Fraction(1)
        for i in range(50):
            mid = (l_pa + r_pa) / 2
            if self.check_radius_binary(poisoned_ins_num, poisoned_feat_num, mid):
                r_pa = mid
            else:
                l_pa = mid

        self.pa_lb_cache[(poisoned_ins_num, poisoned_feat_num)] = r_pa
        np.save(self.cache_file, self.pa_lb_cache)
        return r_pa
