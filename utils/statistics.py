import random

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency

from typing import Sequence, Tuple


def chi_square_test_train(
    train_data: pd.DataFrame, test_data: pd.DataFrame, significance_level: float = 0.05
) -> None:
    """
    Calculates chi-squared statistics for a train and test dataframe.
    :param train_data: training data in a dataframe
    :param test_data: testing data in a dataframe
    :param significance_level: deafult significance level 0.05, can be set arbitrarily
    :return None:
    """
    train_total = train_data.shape[0]
    test_total = test_data.shape[0]
    rel_freq_train = (
        train_data["quality"].value_counts().apply(lambda x: x / train_total)
    )
    rel_freq_test = test_data["quality"].value_counts().apply(lambda x: x / test_total)
    combined_data = pd.merge(rel_freq_train, rel_freq_test, on="quality", how="left")
    chi2, p_value, dof, expected = chi2_contingency(combined_data)
    print("Chi-squared statistic:", chi2)
    print("P-value:", p_value)

    alpha = significance_level
    if p_value < alpha:
        print(
            "Reject null hypothesis: There is a significant difference in the distribution of the categorical variable between the two datasets."
        )
    else:
        print(
            "Fail to reject null hypothesis: There is no significant difference in the distribution of the categorical variable between the two datasets."
        )


def bootstrap_confidence_interval_two_means(
    obs1: pd.Series, obs2: pd.Series, alpha: float = 0.05, n_bootstrap: int = 1000
) -> tuple:
    """
    Calculate the bootstrap confidence interval for the difference between two means
    Parameters:
    - obs1 (pd.Series): dataset number 1
    - obs2 (pd.Series): dataset number 2
    - alpha (float): desired significance level
    - n_bootstrap (int): number of bootstrap samples

    Returns:
    - lower_bound (float): The lower bound of the confidence interval
    - upper_bound (float): The upper bound of the confidence
    """
    n_obs1, n_obs2 = len(obs1), len(obs2)

    bootstrap_means_diff = []
    for _ in range(n_bootstrap):
        bootstrap_sample1 = np.random.choice(obs1, size=n_obs1, replace=True)
        bootstrap_sample2 = np.random.choice(obs2, size=n_obs2, replace=True)

        mean1 = np.mean(bootstrap_sample1)
        mean2 = np.mean(bootstrap_sample2)

        means_diff = mean1 - mean2
        bootstrap_means_diff.append(means_diff)

    lower_bound = np.percentile(bootstrap_means_diff, 100 * alpha / 2)
    upper_bound = np.percentile(bootstrap_means_diff, 100 * (1 - alpha / 2))

    return lower_bound, upper_bound


def mean_diff_permutation(values: Sequence, n_obs_a: int, n_obs_b: int) -> float:
    """
    Calculate the mean difference for a single permutation of data from two groups of observations.

    Parameters:
    - values (Sequence): a Sequence such as a pandas series or list of values for all observations from
                         two independent groups.
    - n_obs_a (int): The number of observations in group A.
    - n_obs_b (int): The number of observations in group B.

    Returns:
    - float: The mean difference for a single permutation of the data from two groups of observations.
    """
    total_obs = n_obs_a + n_obs_b
    idx_a = set(random.sample(range(total_obs), n_obs_a))
    idx_b = set(range(total_obs)) - idx_a
    return values.iloc[list(idx_a)].mean() - values.iloc[list(idx_b)].mean()
