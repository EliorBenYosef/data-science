"""
a Multi-Armed Bandit (MAB) problem
- slot-machines selection:
        you have multiple slot-machines,
        each with a different win distribution,
        you want to find the one with the best (highest) distribution.
- ads selection: you're an advertiser, and you want to find the best ad.
        you have multiple ads,
        each with a different Clickthrough Rate (CTR) distribution (r_i(n) ∈ {0,1}),
        you want to find the one with the best (highest) distribution = Ads CTR Optimization.

this can also be solved with A/B testing (running an A/B test), which is pure exploration
        (you exploit all the options the same), which is less efficient.
"""

import matplotlib.pyplot as plt
from algorithms_mab import ucb, ts
import random
import numpy as np
from utils import plot_hist_sum, plot_hist_count

D = 10  # the problem dimensionality (number of arms = ads).

dist_p = []  # the probabilities of Binomial Distributions of the arms
for i in range(D):
    dist_p.append(random.uniform(0, 1))
dist_p = np.array(dist_p)
np.set_printoptions(precision=3)
print(dist_p)

N = 200  # number of experiments \ rounds \ iterations

ads_selected_ucb = ucb(dist_p, D, N)
ads_selected_ts = ts(dist_p, D, N)

#########################################

# plot Histogram:

mab_algorithms = ['Upper Confidence Bound', 'Thompson Sampling']
data = np.array([ads_selected_ucb, ads_selected_ts])

xlabel = 'Ads'
ylabel = 'Number of selections'
title = f'Histogram of Ads Selections\np = {dist_p}'
# bars_type = 'adjacent'
bars_type = 'overlapping'

# # Histogram of count (ads_selected):
# plot_hist_count(data, mab_algorithms, ylabel, xlabel, title, adjacent_bars=bars_type == 'adjacent')
# Histogram of sum (n_selections) = Bars:
plot_hist_sum(data, mab_algorithms, ylabel, xlabel, title, adjacent_bars=bars_type == 'adjacent')

plt.savefig(f'results/MAB_algo_{bars_type}.png')
plt.show()
