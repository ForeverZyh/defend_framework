from fractions import Fraction

from utils import preprocessing_counts


def check_radius_binary(x, pa, d, Ia, Ib, K, early_stop=None):
    remain_to_assign = pa
    achieved = Fraction(0)
    for m_n_delta in range(-d, d + 1):
        q_delta = 0
        p_delta = 0
        for m in range(max(0, m_n_delta), min(d, d + m_n_delta)):
            n = m - m_n_delta
            if 0 <= n <= d:
                if n >= m:
                    c = preprocessing_counts.get_count(d, m, n, x, K)
                else:
                    c = preprocessing_counts.get_count(d, n, m, x, K)
                if c == 0:
                    continue
                p_delta += c * preprocessing_counts.my_powe(Ia, d - m) * preprocessing_counts.my_powe(Ib, m)
                q_delta += c * preprocessing_counts.my_powe(Ia, d - n) * preprocessing_counts.my_powe(Ib, n)

        if p_delta == 0 and q_delta == 0:
            continue

        if p_delta < remain_to_assign:
            remain_to_assign -= p_delta
            achieved += q_delta
            if early_stop is not None and achieved > early_stop:
                return achieved
        else:
            achieved += remain_to_assign / ((Ia ** (-m_n_delta)) * (Ib ** m_n_delta))
            return achieved

    return achieved


if __name__ == "__main__":
    d = (28 * 28 + 1) * 100 + 28 * 28
    K = 1
    Ia = Fraction(9, 10)
    Ib = (Fraction(1) - Ia) / K
    poisoned_ins_num = 1  # total flips

    l_pa = Fraction(1, 2)
    r_pa = Fraction(1)
    for i in range(13):  # about 1e-4
        mid = (l_pa + r_pa) / 2
        if check_radius_binary(poisoned_ins_num, mid, d, Ia, Ib, K, early_stop=Fraction(1, 2)):
            r_pa = mid
        else:
            l_pa = mid
        print(float(r_pa))
