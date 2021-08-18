"""
Association Rule Learning (ARL) models
learning correlations (relationships):
People who did something also did something else...

Run the following command in the terminal to install the apyori package: pip install apyori

#############################

Apriori -> rules (LHS -> RHS)
a slow algorithm. goes over all the recommendation permutations.
more recommended than Eclat.

Item recommendation: I - an item.
Support(I) = # objects containing I / # objects
Confidence(I1 -> I2) = # objects containing I1 and I2 / # objects containing I1
Lift(I1 -> I2) = Confidence(I1 -> I2) / Support(I2)
the lift is the improvement in the prediction

Algorithm steps:
1. set a minimum support and confidence.
2. take all the subsets in transactions having higher support than minimum support.
3. take all the rules of these subsets having higher confidence than minimum confidence.
4. sort the rules by decreasing lift

#############################

Eclat -> sets (Item1, Item2, ...).
kind of like a simplified version of Apriori.
here we only have support.
a very trivial approach.

Item recommendation: I - a set of (2+) items.
Support(I) = # objects containing I / # objects
how frequently does this set of items occur.

Algorithm steps:
1. set a minimum support.
2. take all the subsets in transactions having higher support than minimum support.
4. sort these subsets by decreasing support
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from apyori import apriori

df = pd.read_csv(
    '../../../datasets/per_field/usl/association_rule_learning/Market_Basket.csv',
    header=None)

# min_support: (3 transactions/day) * (7 days/week) / (7501 transactions/week) = 0.0028 -> 0.003
# min_confidence - try different values depending on the business requirements, and see how many rules you get.
#   the default in the R library is 0.8, but 0.8 didn't result in any rule, and 0.4 didn't result in enough rules.
# min_lift - less than 3 is not good enough. 3 is the minimum.
rules_gen = apriori(transactions=df.values.astype(np.str),
                    min_support=0.003, min_confidence=0.2, min_lift=3,
                    min_length=2, max_length=2)  # rules containing only 2 products (for the "buy 1 get 1 free" purpose)
rules_list = list(rules_gen)


def rules_list_to_df(rules):
    rules_neat = [[
        tuple(rule[2][0][0])[0],    # Left Hand Side (LHS), items_base
        tuple(rule[2][0][1])[0],    # Right Hand Side (RHS), items_add
        rule[1],                    # Supports
        rule[2][0][2],              # Confidences
        rule[2][0][3]               # Lifts
    ] for rule in rules]
    return pd.DataFrame(rules_neat, columns=['LHS', 'RHS', 'Support', 'Confidence', 'Lift'])


rules_df = rules_list_to_df(rules_list)

# sort rules by descending lifts:
print('Apriori -> rules (LHS -> RHS)\n',
      rules_df.nlargest(n=10, columns='Lift'), '\n')

# sort rules by descending supports:
print('Eclat -> sets (RHS, LHS)\n',
      rules_df[['LHS', 'RHS', 'Support']].nlargest(n=10, columns='Support'))
