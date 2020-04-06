import pandas as pd
import numpy as np
import skfuzzy as fuzz
from scipy import stats
from skfuzzy import control as ctrl
from sklearn.datasets import make_blobs, make_classification, load_iris, load_digits
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeClassifier

from fuzzyml import FuzzyClassifier
from fuzzyml import plot_decision_function
from fuzzyml import extract_rules


def main():
    # X, y = make_classification(n_samples=1000, n_features=2, n_redundant=0, n_informative=1,
    #                            n_clusters_per_class=1, random_state=0)
    X, y = load_iris(return_X_y=True)
    X = pd.DataFrame(X)
    cols = [f"F{i}" for i in range(0, X.shape[1])]
    X.columns = cols

    fuzzy_scores = []
    standard_scores = []

    skf = StratifiedKFold(n_splits=10)
    i = 0
    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y[train_index], y[test_index]

        rules = extract_rules(X_train, y_train, alpha=0.01, verbose=False)

        # control
        ctrl_sys = ctrl.ControlSystem(rules)
        fuzzy_ctrl = ctrl.ControlSystemSimulation(ctrl_sys)

        # load ML models
        model = DecisionTreeClassifier(random_state=0)
        fuzzy_model = FuzzyClassifier(model, fuzzy_ctrl, alpha=0.5)

        # fit fuzzy model
        fuzzy_model.fit(X_train, y_train)
        score_fz = fuzzy_model.score(X_test, y_test)

        # fit standard model
        model.fit(X_train, y_train)
        score_dt = model.score(X_test, y_test)

        fuzzy_scores.append(score_fz)
        standard_scores.append(score_dt)

        if X.shape[1] == 2:
            plt.figure(figsize=[10, 5])
            plt.subplot(121)
            plot_decision_function(fuzzy_model, "Fuzzy - acc %.2f" % score_fz, X, y, train_index)
            plt.subplot(122)
            plot_decision_function(model, "No rules - acc %.2f" % score_dt, X, y, train_index)
            plt.tight_layout()
            plt.savefig(f"./cv_{i:2d}.png")
            plt.show()

        i += 1

    print(f"Standard ML model: {np.mean(standard_scores):.4f} +- ({np.std(standard_scores)**2:.4f})")
    print(f"ML model + extracted rules: {np.mean(fuzzy_scores):.4f} +- ({np.std(fuzzy_scores)**2:.4f})")

    return


if __name__ == '__main__':
    main()
