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

d = 10  # the problem dimensionality (number of arms = ads).

dist_p = []  # the probabilities of Binomial Distributions of the arms
for i in range(d):
    dist_p.append(random.uniform(0, 1))
dist_p = np.array(dist_p)
np.set_printoptions(precision=3)
print(dist_p)

N = 200  # number of experiments \ rounds \ iterations

ads_selected_ucb = ucb(dist_p, d, N)
ads_selected_ts = ts(dist_p, d, N)

#########################################

# plot Histogram:

# bins = np.linspace(1, d, d)
bins = np.arange(d) + 1

fig, ax = plt.subplots(figsize=(10, 6))

# # Histogram of count (ads_selected):
# # Option 1 (adjacent bars):
# ax.hist([ads_selected_ucb, ads_selected_ts], bins,
#         label=['Upper Confidence Bound', 'Thompson Sampling'])
# # Option 2 (overlapping\superimposed bars):
# ax.hist(ads_selected_ucb, bins, alpha=0.5, label='Upper Confidence Bound')
# ax.hist(ads_selected_ts, bins, alpha=0.5, label='Thompson Sampling')

# Histogram of sum (n_selections) = Bars:
# Option 1 (adjacent bars):
ax.bar(bins - 0.2, ads_selected_ucb, width=0.4, align='center', label='Upper Confidence Bound')
ax.bar(bins + 0.2, ads_selected_ts, width=0.4, align='center', label='Thompson Sampling')
# # Option 2 (overlapping\superimposed bars):
# ax.bar(bins, ads_selected_ucb, alpha=0.75, width=0.8, align='center', label='Upper Confidence Bound')
# ax.bar(bins, ads_selected_ts, alpha=0.75, width=0.8, align='center', label='Thompson Sampling')

ax.legend(loc='upper left')
ax.set_xlabel('Ads')
ax.set_ylabel('Number of selections')
ax.set_xticks(bins)
ax.set_title(f'Histogram of Ads Selections\np = {dist_p}')

plt.show()
