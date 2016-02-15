import numpy as np
import mpmath
from mpmath import bernpoly, mpf, fprod

__ONE_OVER_LOG10 = 0.43429448190325176
mpmath.mp.dps = 500


def exp_series(A):
    B = [1]
    for k in range(1, len(A)):
        S = sum(j * A[j] * B[k - j] / k for j in range(1, k + 1))
        B.append(S)
    return B


def gammaprod_series(As, Bs, M):
    A = [0] + [sum((-1) ** k * (bernpoly(k, As[j]) - bernpoly(k, Bs[j
                                                                    ])) / (k * (k - 1))
                   for j in range(len(As))) for k in range(2, M + 1)]
    return exp_series(A)


def hyper1(As, Bs, N, M):
    with mpmath.extraprec(mpmath.mp.prec):
        s = t = 1
        for j in range(1, N):
            t *= fprod(a + j - 1 for a in As) / fprod(b + j - 1 for b in Bs)
            s += t
        if M > 0:
            s2 = 0
            g = sum(As) - sum(Bs)
            for (j, c) in enumerate(gammaprod_series(As, Bs, M)):
                s2 += c * mpmath.zeta(-(g - j), N)
            s += s2 * mpmath.gammaprod(Bs, As)
    return s


def hyper1_auto(As, Bs, N, M):
    with mpmath.extraprec(mpmath.mp.prec):
        s = t = 1
        good_ratio_hits = 0
        for j in range(1, N):
            s_old = s
            t *= fprod(a + j - 1 for a in As) / fprod(b + j - 1 for b in Bs)
            s += t
            ratio = (s - s_old) / s
            if ratio < mpf(10 ** -18):
                good_ratio_hits += 1
            if good_ratio_hits > 3:
                break
            print float(s)
        if M > 0:
            s2 = 0
            g = sum(As) - sum(Bs)
            for (j, c) in enumerate(gammaprod_series(As, Bs, M)):
                s2 += c * mpmath.zeta(-(g - j), N)
            s += s2 * mpmath.gammaprod(Bs, As)
    return s


def sum_log_range(minimum, maximum):
    return np.log(np.arange(minimum, maximum + 1)).sum()


def log_hyper_probability(F, M, n, j):
    logH = log_binomial(
        M, j) + log_binomial(F - M, n - j) - log_binomial(F, n)
    return logH * __ONE_OVER_LOG10


def log_binomial(n, k):
    if k == n or not k:
        return 0
    elif n > 1000 and k > 1000:  # Stirling's binomial coeff approximation
        return n * np.log(n) - (n - k) * np.log(n - k) - k * np.log(k)
    else:
        t = n - k
        if t < k:
            t = k
        return sum_log_range(t + 1, n) - sum_factorial(n - t)


def sum_factorial(n):
    if n > 1000:
        return n * np.log(n) - n
    else:
        return sum_log_range(2, n)


def sum_log_probabilities(next_log_P, log_P):
    if next_log_P == 0:
        return True, log_P
    # Several optimizations to avoid over/underflow problems
    common, diffExponent = 0.0, 0.0
    if next_log_P > log_P:
        common = next_log_P
        diffExponent = log_P - common
    else:
        common = log_P
        diffExponent = next_log_P - common
    log_P = common + \
        ((np.log(1.0 + 10.0 ** diffExponent)) * __ONE_OVER_LOG10)
    # The cumulative summation stops when the increasing is less than 10e-4
    if (next_log_P - log_P) < -4.0:
        return True, log_P
    return False, log_P


def compute_surprise(mi, pi, m, p):
    j = mi
    log_p = log_hyper_probability(p, pi, m, mi)
    minimum = pi
    if m < pi:
        minimum = m
    is_enough = False
    while not is_enough and j < minimum:
        j += 1
        next_log_p = log_hyper_probability(p, pi, m, j)
        is_enough, log_p = sum_log_probabilities(next_log_p, log_p)
    if log_p == 0:
        log_p *= -1
    return -log_p
