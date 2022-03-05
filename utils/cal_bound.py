from abc import ABC, abstractmethod
from fractions import Fraction
import os
import numpy as np
from scipy.special import comb

from utils.preprocessing_counts import process_count


class BoundCalculator(ABC):
    def __init__(self):
        self.pa_lb_cache = dict()
        self.cache_file = None
        self.k = None
        self.D = None

    @abstractmethod
    def cal_NP_bound(self, s, remain_to_assign, p_binom, reverse=False, early_stop=None):
        pass

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


class SelectBoundCalculator(BoundCalculator):
    def __init__(self, IaIbK, dataset, D, L, k, l):
        super(SelectBoundCalculator, self).__init__()
        if IaIbK is not None:
            self.Ia, self.Ib, self.K = IaIbK
        else:  # if we do not flip the label
            self.Ia, self.Ib, self.K = None, None, None
        self.k = k

        self.fn = dataset
        if not os.path.exists(os.path.join("list_counts", dataset)):
            os.mkdir(os.path.join("list_counts", dataset))

        self.D = D
        self.L = L
        self.l = l
        self.cache_file = os.path.join("list_counts", dataset,
                                       f"cache_{str(IaIbK).replace('/', '__')}_{k}_{L}_{l}")
        try:
            self.pa_lb_cache = np.load(self.cache_file + ".npy", allow_pickle=True).item()
        except:
            pass

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
        Ia, Ib, fn, D, L, K, k, l = self.Ia, self.Ib, self.fn, self.D, self.L, self.K, self.k, self.l
        achieved = 0
        fact = [1]
        for i in range(1, L + s + 1):
            fact.append(fact[-1] * i)
        coefs = [Fraction(1)]
        for i in range(1, k + 1):
            coefs.append(coefs[-1] * Fraction(fact[L] * fact[L + s - l], fact[L - l] * fact[L + s]))

        outcome = [(1, 1, 1, 0)]
        for i in range(1, k + 1):
            outcome.append((1 - coefs[i], 0, np.inf, i))
        if Ia is None:  # no label flipping
            for i in range(1, k + 1):
                outcome.append((coefs[i], 1, coefs[i], i))
        else:
            complete_cnt_ps = [[1], [Ib, Ib * (K - 1), Ia]]
            complete_cnt_qs = [[1], [Ia, Ib * (K - 1), Ib]]
            for i in range(2, k + 1):
                complete_cnt_ps.append([0] * (i * 2 + 1))
                complete_cnt_qs.append([0] * (i * 2 + 1))
                for j in range(3):
                    for k_ in range((i - 1) * 2 + 1):
                        complete_cnt_ps[i][j + k_] += complete_cnt_ps[1][j] * complete_cnt_ps[i - 1][k_]
                        complete_cnt_qs[i][j + k_] += complete_cnt_qs[1][j] * complete_cnt_qs[i - 1][k_]
            for i in range(1, k + 1):
                for j in range(i * 2 + 1):
                    p = complete_cnt_ps[i][j] * coefs[i]
                    outcome.append((p, complete_cnt_qs[i][j], p / complete_cnt_qs[i][j], i))

        outcome.sort(key=lambda x: x[2] if reverse else -x[2])
        for i in range(len(outcome)):
            p_cnt, q_cnt, eta, poison_cnt = outcome[i]

            p_delta = p_cnt * p_binom[poison_cnt]
            q_delta = q_cnt * p_binom[poison_cnt]

            if p_delta < remain_to_assign:
                remain_to_assign -= p_delta
                achieved += q_delta
                if early_stop is not None and achieved > early_stop:
                    return achieved
            else:
                achieved += remain_to_assign / eta
                return achieved
        return achieved


class FlipBoundCalculator(BoundCalculator):
    def __init__(self, Ia, Ib, dataset, D, d, K, k):
        super(FlipBoundCalculator, self).__init__()
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
            pass

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
        filename = os.path.join("list_counts", self.fn, f'{run_name}.npz')
        if os.path.exists(filename):
            npzfile = np.load(filename, allow_pickle=True)
            complete_cnt_ps, complete_cnt_qs = npzfile["complete_cnt_ps"], npzfile["complete_cnt_qs"]
        else:
            print("preparing " + run_name)
            complete_cnt_ps = [[1], complete_cnt_p]
            complete_cnt_qs = [[1], complete_cnt_q]
            # compute convolutions
            for i in range(2, k + 1):
                complete_cnt_ps.append([0] * (i * d * 2 + 1))
                complete_cnt_qs.append([0] * (i * d * 2 + 1))
                for j in range(d * 2 + 1):
                    if complete_cnt_ps[1][j] > 0 or complete_cnt_qs[1][j] > 0:
                        for k_ in range(d * (i - 1) * 2 + 1):
                            complete_cnt_ps[i][j + k_] += complete_cnt_ps[1][j] * complete_cnt_ps[i - 1][k_]
                            complete_cnt_qs[i][j + k_] += complete_cnt_qs[1][j] * complete_cnt_qs[i - 1][k_]
            np.savez(filename,
                     complete_cnt_ps=complete_cnt_ps,
                     complete_cnt_qs=complete_cnt_qs)
            print("save file " + run_name + ".npz")

        if not reverse:
            _range = range(-k * d, k * d + 1)
        else:
            _range = range(k * d, -k * d - 1, -1)

        for m_n_delta in _range:
            outcome = []
            for i in range(k + 1):
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
