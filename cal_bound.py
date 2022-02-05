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
    def __init__(self, Ia, Ib, dataset, D, d, K, k, considered_degree=2, algorithm="NP+NP", comp_bound_timeout=10000):
        self.considered_degree = considered_degree
        assert self.considered_degree <= 2
        self.Ia = Ia
        self.Ib = Ib
        self.K = K
        self.k = k
        self.comp_bound_timeout = comp_bound_timeout
        if algorithm == "NP+NP" and comp_bound_timeout < ((k * d) ** 3) * k:
            algorithm = "NP"
        self.algorithm = algorithm

        self.fn = dataset
        if not os.path.exists(os.path.join("list_counts", dataset)):
            os.mkdir(os.path.join("list_counts", dataset))

        self.D = D
        self.d = d
        self.cache_file = os.path.join("list_counts", dataset,
                                       f"cache_{str(Ia).replace('/', '__')}_{str(Ib).replace('/', '__')}_{K}_{k}_{d}_"
                                       f"{algorithm}_{considered_degree}")
        try:
            self.pa_lb_cache = np.load(self.cache_file + ".npy", allow_pickle=True).item()
        except:
            self.pa_lb_cache = dict()

    def check_NP_binary(self, s, pa, p_binom):
        return self.cal_NP_bound(s, pa, p_binom, early_stop=Fraction(1, 2)) > Fraction(1, 2)

    def check_NP_NP_binary(self, s, pa, p_binom):
        return self.cal_NP_NP_bound(s, pa, p_binom, early_stop=Fraction(1, 2)) > Fraction(1, 2)

    def check_NP(self, s, pa, pb, _1mp1, p_binom):
        """
        :param s: the number of poisoned features
        :param pa:
        :param pb:
        :param _1mp1: the likelihood of L^{l+1} ... L^{k} that we won't compute
        :param p_binom:
        :return: whether it is certifiable
        """
        lb = min(self.cal_NP_bound(s, pb, p_binom, reverse=True, early_stop=Fraction(1, 2) - _1mp1) + _1mp1,
                 Fraction(1, 2))
        return self.cal_NP_bound(s, pa - _1mp1, p_binom, early_stop=lb) > lb

    def check_NP_NP(self, s, pa, pb, p_binom):
        """
        :param s: the number of poisoned features
        :param pa:
        :param pb:
        :param p_binom:
        :return: whether it is certifiable
        """
        lb = min(self.cal_NP_NP_bound(s, pb, p_binom, reverse=True, early_stop=Fraction(1, 2)), Fraction(1, 2))
        return self.cal_NP_NP_bound(s, pa, p_binom, early_stop=lb) > lb

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
        considered_degree = self.considered_degree
        achieved = 0

        complete_cnt_p, complete_cnt_q = process_count(Ia, Ib, d, K, s)
        if not reverse:
            _range = range(-considered_degree * d, considered_degree * d + 1)
        else:
            _range = range(considered_degree * d, -considered_degree * d - 1, -1)

        for m_n_delta in _range:
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
                    if early_stop is not None and achieved > early_stop:
                        return achieved
                else:
                    achieved += remain_to_assign / ((Ia ** (-m_n_delta)) * (Ib ** m_n_delta))
                    return achieved

        return achieved

    def cal_NP_NP_bound(self, s, remain_to_assign, p_binom, reverse=False, early_stop=None):
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
        considered_degree = self.considered_degree
        achieved = 0

        complete_cnt_p, complete_cnt_q = process_count(Ia, Ib, d, K, s)
        complete_cnt_ps = [None] * (k + 1)
        complete_cnt_qs = [None] * (k + 1)

        for i in range(considered_degree + 1, k + 1):
            complete_cnt_ps[i], complete_cnt_qs[i] = process_count(Ia, Ib, d * k, K, s * i)

        if not reverse:
            _range = range(-d * k, d * k + 1)
        else:
            _range = range(d * k, -d * k - 1, -1)

        for m_n_delta in _range:
            outcome = []
            if -considered_degree * d <= m_n_delta <= considered_degree * d:
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

            for i in range(considered_degree + 1, k + 1):
                outcome.append([complete_cnt_ps[i][m_n_delta + d * k], complete_cnt_qs[i][m_n_delta + d * k], i])

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

    def check_NP_KL_binary(self, s, r, p, _1mp1, p_binom):
        Ia, Ib, fn, D, d, K, k = self.Ia, self.Ib, self.fn, self.D, self.d, self.K, self.k
        considered_degree = self.considered_degree
        p1 = 1 - _1mp1
        achieved = 0
        assigned = 0

        complete_cnt_p, complete_cnt_q = process_count(Ia, Ib, d, K, s)

        for m_n_delta in range(-considered_degree * d, considered_degree * d + 1):
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
                    def get_value(mid):
                        if _1mp1 == Fraction(0):
                            return achieved + (mid - assigned) / ratio
                        return achieved + (mid - assigned) / ratio + _1mp1 * get_KL_divergence_cf(
                            (p - mid) / _1mp1,
                            ((k - considered_degree) * Fraction(r, D) + considered_degree) * log(Ia / Ib) * s * (
                                    Ia - Ib))

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
                    return True

        return True

    def check_radius_binary(self, x, s, pa):
        """
        return whether the radius is certifiable
        :param x: the number of poisoned instance
        :param s: the number of poisoned feature
        :param pa: the lower bound of the probability of the most likely label
        :return: bool, whether it is certifiable
        """
        p_binom = [None] * (self.k + 1)
        other_pk = Fraction(1)
        for i in range(min(self.considered_degree, self.k) + 1):
            p_binom[i] = comb(self.k, i, exact=True) * (Fraction(x, self.D) ** i) * (
                    (1 - Fraction(x, self.D)) ** (self.k - i))
            other_pk -= p_binom[i]

        if self.algorithm == "NP":
            return self.check_NP_binary(s, Fraction(pa) - other_pk, p_binom)
        elif self.algorithm == "NP+NP":
            for i in range(self.considered_degree + 1, self.k + 1):
                p_binom[i] = comb(self.k, i, exact=True) * (Fraction(x, self.D) ** i) * (
                        (1 - Fraction(x, self.D)) ** (self.k - i))
            # instead of subtract other_pk, we will look them into details
            return self.check_NP_NP_binary(s, Fraction(pa), p_binom)
        elif self.algorithm == "NP+KL":
            return self.check_NP_KL_binary(s, x, Fraction(pa), other_pk, p_binom)
        else:
            raise NotImplementedError

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
        other_pk = Fraction(1)
        for i in range(self.considered_degree + 1):
            p_binom[i] = comb(self.k, i, exact=True) * (Fraction(x, self.D) ** i) * (
                    (1 - Fraction(x, self.D)) ** (self.k - i))
            other_pk -= p_binom[i]

        if self.algorithm == "NP":
            return self.check_NP(s, Fraction(pa), Fraction(pb), other_pk, p_binom)
        elif self.algorithm == "NP+KL":
            raise NotImplementedError
        elif self.algorithm == "NP+NP":
            for i in range(self.considered_degree + 1, self.k + 1):
                p_binom[i] = comb(self.k, i, exact=True) * (Fraction(x, self.D) ** i) * (
                        (1 - Fraction(x, self.D)) ** (self.k - i))
            # instead of subtract other_pk, we will look them into details
            return self.check_NP_NP(s, Fraction(pa), Fraction(pb), p_binom)
        else:
            raise NotImplementedError

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
