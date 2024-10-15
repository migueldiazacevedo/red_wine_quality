import matplotlib.pyplot as plt
import matplotlib.figure
import numpy as np
from numpy.typing import ArrayLike
import pandas as pd
import seaborn as sns
import scipy.stats as stats

from typing import List, Tuple


def plot_correlations(
    df: pd.DataFrame, title: str, annot: bool = True, fmt: str = ".2f"
) -> Tuple[matplotlib.figure.Figure, plt.Axes]:
    """
    Plot Correlations between columns in a dataframe.

    Parameters:
    - df (pandas DataFrame): features and outcomes dataframe.
    - title (string): A title for the graph.
    - annot (boolean): True for annotation, False for no annotation
    - fmt (string): A formatting string to use in annotation.

    Returns:
    - fig (matplotlib.figure.Figure): A matplotlib figure.
    - ax (matplotlib.axes.Axes): Axes object for plotting.
    """
    correlations = df.corr()
    mask = np.triu(np.ones_like(correlations, dtype=bool))

    fig, ax = plt.subplots()
    sns.heatmap(correlations, annot=annot, fmt=fmt, cmap="Blues_r", ax=ax, mask=mask)
    ax.set_title(title)
    plt.show()
    return fig, ax


def plot_feature_boxplot(
    features: ArrayLike, df: pd.DataFrame
) -> Tuple[matplotlib.figure.Figure, plt.Axes]:
    """
    Plot boxplot of features in a dataframe.

    Parameters:
    - features (ArrayLike): A List, Array or other grouping of features in dataframe df.
    - df (pandas DataFrame): features and outcomes dataframe.

    Returns:
    - fig (matplotlib.figure.Figure): A matplotlib figure.
    - ax (matplotlib.axes.Axes): Axes object for plotting.
    """
    fig, axes = plt.subplots(nrows=3, ncols=int(len(features) / 3), figsize=(10, 5))
    fig.tight_layout()

    for i, feature in enumerate(features):
        sns.boxplot(y=df[feature], ax=axes.flatten()[i])

    return fig, axes


def plot_feature_distribution(
    features: ArrayLike, df: pd.DataFrame, kde: bool = True
) -> Tuple[matplotlib.figure.Figure, plt.Axes]:
    """
    Plot histogram of features in a dataframe.

    Parameters:
    - features (ArrayLike): A List, Array or other grouping of features in dataframe df.
    - df (pandas DataFrame): features and outcomes dataframe.
    - kde (boolean): True for KDE, False otherwise.

    Returns:
    - fig (matplotlib.figure.Figure): A matplotlib figure.
    - ax (matplotlib.axes.Axes): Axes object for plotting.
    """
    fig, axes = plt.subplots(nrows=4, ncols=int(len(features) / 4), figsize=(10, 5))
    fig.tight_layout()

    for i, feature in enumerate(features):
        sns.histplot(x=df[feature], kde=kde, ax=axes.flatten()[i])

    return fig, axes


def plot_feature_qqplot(
    features: ArrayLike, df: pd.DataFrame
) -> Tuple[matplotlib.figure.Figure, plt.Axes]:
    """
    Plot q-q plot of features in a dataframe for a normal assumption.

    Parameters:
    - features (ArrayLike): A List, Array or other grouping of features in dataframe df.
    - df (pandas DataFrame): features and outcomes dataframe.
    - kde (boolean): True for KDE, False otherwise.

    Returns:
    - fig (matplotlib.figure.Figure): A matplotlib figure.
    - ax (matplotlib.axes.Axes): Axes object for plotting.
    """
    fig, axes = plt.subplots(nrows=3, ncols=int(len(features) / 3), figsize=(12, 6))
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    fig.subplots_adjust(hspace=0.5, wspace=0.3)

    for i, feature in enumerate(features):
        stats.probplot(df[feature], plot=axes.flatten()[i])
        axes.flatten()[i].set_title(feature)
        if i in [8, 9, 10, 11]:
            axes.flatten()[i].set_xlabel("Theoretical quantiles")
        else:
            axes.flatten()[i].set_xlabel("")
        if i in [0, 4, 8]:
            axes.flatten()[i].set_ylabel("Ordered Values")
        else:
            axes.flatten()[i].set_ylabel("")

    plt.subplots_adjust(top=0.9)
    fig.suptitle("Q-Q plots for each feature -- Normal Assumption")

    return fig, axes


def plot_permutations(permutation_diffs: List, best_estimate: float) -> plt.Axes:
    """
    Plot permutations of features in a dataframe.
    Parameters:
    - permutation_diffs (List): permutation mean differences (or differences for any statistic used in permutation)
    - best_estimate (number, float or int): The best estimate of the feature.

    Returns:
    - ax (matplotlib.axes.Axes): Axes object for plotting.
    """
    ax = sns.histplot(data=permutation_diffs, bins=10)
    ax.axvline(x=best_estimate, linestyle="--", color="black")

    sns.despine(left=True, bottom=True, right=True, top=True)
    ax.xaxis.set_ticks_position("bottom")
    ax.xaxis.set_tick_params(width=1, length=5)
    ax.yaxis.set_ticks_position("left")
    ax.yaxis.set_tick_params(width=1, length=5)
    ax.text(
        best_estimate - 0.075, 200, "Observed\ndifference", bbox={"facecolor": "white"}
    )

    plt.title("Permutation Test of Mean Rating Difference")
    plt.xlabel("Mean Rating Difference")
    plt.ylabel("Count")

    return ax


def plot_actual_vs_predicted(actual, predicted):
    plt.scatter(actual, predicted)
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title("Actual vs. Predicted")
    plt.show()


def plot_residuals(predictions, residuals):
    plt.scatter(predictions, residuals)
    plt.xlabel("Predicted")
    plt.ylabel("Residuals")
    plt.title("Residual Plot")
    plt.axhline(y=0, color="r", linestyle="-")
    plt.show()

    sns.histplot(residuals, kde=True)
    plt.xlabel("Residuals")
    plt.ylabel("Frequency")
    plt.title("Histogram of Residuals")
    plt.show()


def plot_roc_curve(fpr, tpr, roc_auc):
    plt.figure()
    plt.plot(
        fpr, tpr, color="darkorange", lw=2, label="ROC curve (area = %0.2f)" % roc_auc
    )
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic")
    plt.legend(loc="lower right")
    plt.show()
