import numpy as np
import matplotlib.pyplot as plt


def plot_hist_sum(data, r_labels, ylabel, xlabel, title, x_tick_labels=None, adjacent_bars=True):
    """
    plots Histogram of sum
    plots multiple bars
    """
    n_rows = len(r_labels)

    total_bars_width = 0.8
    bar_width = total_bars_width / n_rows
    # bins = np.linspace(1, d, d)
    bins = np.arange(data.shape[1]) + 1  # d = data.shape[1]

    fig, ax = plt.subplots(figsize=(10, 6))
    for i in range(n_rows):
        if adjacent_bars:
            ax.bar(x=bins + (i * bar_width) - total_bars_width / 2 + bar_width / 2,
                   height=data[i], width=bar_width, align='center', label=r_labels[i])
        else:  # overlapping\superimposed bars
            ax.bar(x=bins, height=data[i], alpha=0.75, width=total_bars_width, align='center', label=r_labels[i])

    ax.legend(loc='best')
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    ax.set_xticks(bins)
    if x_tick_labels is not None:
        ax.set_xticklabels(x_tick_labels)
    ax.set_title(title)


def plot_hist_count(data, r_labels, ylabel, xlabel, title, x_tick_labels=None, adjacent_bars=True):
    """
    plots Histogram of count
    """
    n_rows = len(r_labels)

    total_bars_width = 0.8
    bar_width = total_bars_width / n_rows
    # bins = np.linspace(1, d, d)
    bins = np.arange(data.shape[1]) + 1  # d = data.shape[1]

    fig, ax = plt.subplots(figsize=(10, 6))
    if adjacent_bars:
        ax.hist(x=data, bins=bins, label=r_labels)
    else:  # overlapping\superimposed bars
        for i in range(n_rows):  # alpha=0.5
            ax.hist(x=data[i], bins=bins, alpha=0.5, label=r_labels[i])

    ax.legend(loc='best')
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    ax.set_xticks(bins)
    if x_tick_labels is not None:
        ax.set_xticklabels(x_tick_labels)
    ax.set_title(title)
