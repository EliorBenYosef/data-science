import numpy as np
import pandas as pd

df = pd.read_csv('../../../datasets/per_field/sl/reg/linreg_simple.txt', header=None, names=['Population', 'Profit'])
X, y = df.iloc[:, :-1].values, df.iloc[:, -1].values
X = np.squeeze(X)
N = len(y)

m = (N * np.sum(X * y) - np.sum(X) * np.sum(y)) / (N * np.sum(X**2) - np.sum(X)**2)
b = (np.sum(y) - m * np.sum(X)) / N

print(f'y = {m:.2f}x + {b:.2f}')
