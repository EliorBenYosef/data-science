"""
Reinforcement Learning is a powerful branch of Machine Learning.
    It is used to solve interacting real-time problems,
        where the data observed up to time t is considered to decide which action to take at time t + 1.
    Desired outcomes provide the algorithm with reward, undesired with punishment.
    learning is done through trial and error.
"""

import numpy as np
from scipy.stats import binom


def ucb(dist_p, D, N):
    """
    Upper Confidence Bound (UCB) algorithm.
    Deterministic - choosing the ad with max UCB.

    At each round n, for each ad i, we:
    1. Compute:
        • The average r of ad i, up to round n:
                avg.r_i(n) = (R_i(n))/(N_i(n))
            N_i(n) - the number of times the ad i was selected, up to round n,
            R_i(n) - the sum of rewards of the ad i, up to round n.
        • The confidence interval, at round n:
                [avg.r_i(n) - ∆_i(n), avg.r_i(n) + ∆_i(n)]
                ∆_i(n) = √(3∙log(n)/(2∙N_i(n))
    2. Select the ad i that has the maximum UCB:
            UCB = avg.r_i(n) + ∆_i(n).

    :param dist_p: the probabilities of Binomial Distributions of the arms
    :param D: the problem dimensionality (number of arms = ads).
    :param N: number of experiments / rounds / iterations
    :return: n_selections - numbers of selections
    """
    n_selections = np.zeros(D, dtype=int)  # numbers of selections
    r_total = 0

    r_sums = np.zeros(D, dtype=int)  # sums of rewards
    choose_first = 0

    for n in range(1, N+1):
        # choose ad to present:
        if choose_first < D:
            ad = choose_first
            choose_first += 1
        else:
            r_avg = r_sums / n_selections
            delta = np.sqrt(3 / 2 * np.log(n) / n_selections)
            ucb = r_avg + delta
            ad = np.argmax(ucb)

        n_selections[ad] += 1

        # the "environment's response": get a single user interaction with the selected ad
        r = binom.rvs(n=1, p=dist_p[ad], size=1)
        r_total += r

        r_sums[ad] += r

    print(f'UCB total rewards: {r_total}')

    return n_selections


def ts(dist_p, D, N):
    """
    Thompson Sampling algorithm.
    Probabilistic - it constructs a β distribution of the return for each ad,
        samples a value (expected return μ_hat) from each distribution,
        and chooses the ad with max expected return.

    At each round n, for each ad i, we:
    1. Construct a Beta (β) distribution of the return for each ad,
        where the distribution’s mean (μ^*) is the ad’s expected return.
        Sample each distribution to get a predicted expected return (μ ̂ / θ_i(n))
            θ_i(n) = β(N_i^r1(n) + 1, N_i^r0(n) + 1)
        N_i^1(n) - the number of times the ad i got reward 1, up to round n.
        N_i^0(n) - the number of times the ad i got reward 0, up to round n.
    2. Select the ad i that has the highest θ_i(n).

    :param dist_p: the probabilities of Binomial Distributions of the arms
    :param D: the problem dimensionality (number of arms = ads).
    :param N: number of experiments / rounds / iterations
    :return: n_selections - numbers of selections
    """
    n_selections = np.zeros(D, dtype=int)  # numbers of selections

    r_total = 0

    n_r1 = np.zeros(D, dtype=int)  # numbers of rewards 1
    n_r0 = np.zeros(D, dtype=int)  # numbers of rewards 0

    for n in range(N):
        # choose ad to present:
        mu_hat = np.random.beta(n_r1 + 1, n_r0 + 1)  # sampled expected returns
        ad = np.argmax(mu_hat)

        n_selections[ad] += 1

        # the "environment's response": get a single user interaction with the selected ad
        r = binom.rvs(n=1, p=dist_p[ad], size=1)
        r_total += r

        if r == 1:
            n_r1[ad] += 1
        else:
            n_r0[ad] += 1

    print(f'TS total rewards: {r_total}')

    return n_selections
