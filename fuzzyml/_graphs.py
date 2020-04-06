import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_decision_function(classifier, title, X, y, not_visible, const=False):
    # plot the decision function
    xx, yy = np.meshgrid(np.linspace(X.iloc[:, 0].min(), X.iloc[:, 0].max(), 100),
                         np.linspace(X.iloc[:, 1].min(), X.iloc[:, 1].max(), 100))

    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, x_max]x[y_min, y_max].
    x_map = pd.DataFrame(np.c_[xx.ravel(), yy.ravel()])
    x_map.columns = X.columns
    if hasattr(classifier, "decision_function"):
        Z = classifier.decision_function()
    else:
        Z = classifier.predict_proba(x_map)[:, 1]
    Z = Z.reshape(xx.shape)

    # plot the line, the points, and the nearest vectors to the plane
    plt.contourf(xx, yy, Z, alpha=0.75, cmap=plt.cm.bone)
    plt.scatter(X.iloc[~not_visible, 0], X.iloc[~not_visible, 1], c=y[~not_visible], s=100, alpha=0.9,
                 cmap=plt.cm.bone, edgecolors='black',
                label="visible")
    plt.scatter(X.iloc[not_visible, 0], X.iloc[not_visible, 1], c="w",
                marker="s", s=100, alpha=0.9, edgecolors='black',
                label="not visible")

    # plt.plt('off')
    plt.title(title)
