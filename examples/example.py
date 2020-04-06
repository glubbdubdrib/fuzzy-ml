import pandas as pd
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
from sklearn.datasets import make_blobs
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

from fuzzyml import FuzzyClassifier
from fuzzyml import plot_decision_function


def main():

    X, y = make_blobs(n_samples=1000, centers=4, random_state=0)
    X = pd.DataFrame(X)
    cols = ["f1", "f2"]
    X.columns = cols
    y[y == 2] = 0
    not_visible = y == 3
    y[not_visible] = 1
    X_visible = X.iloc[~not_visible]
    y_visible = y[~not_visible]

    # New Antecedent/Consequent objects hold universe variables and membership functions
    feature1 = ctrl.Antecedent(np.arange(-6, 6, 1), X.columns[0])
    feature1['G1'] = fuzz.gaussmf(feature1.universe, -2, 2)
    feature1['G2'] = fuzz.gaussmf(feature1.universe, 2, 2)
    feature1.view()
    feature2 = ctrl.Antecedent(np.arange(-4, 12, 1), X.columns[1])
    feature2['G1'] = fuzz.sigmf(feature2.universe, 2, -2)
    feature2['G2'] = fuzz.gaussmf(feature2.universe, 3, 2)
    feature2['G3'] = fuzz.sigmf(feature2.universe, 5, 2)
    feature2.view()
    label = ctrl.Consequent(np.arange(0, 11, 1), 'label')
    label['L1'] = fuzz.gaussmf(label.universe, 0, 2)
    label['L2'] = fuzz.gaussmf(label.universe, 10, 2)
    label.view()

    # rules
    rule1 = ctrl.Rule(feature1['G1'] & feature2['G2'], label['L2'])
    rule3 = ctrl.Rule(feature2['G3'], label['L1'])

    # control
    ctrl_sys = ctrl.ControlSystem([rule1, rule3])
    fuzzy_ctrl = ctrl.ControlSystemSimulation(ctrl_sys)

    # load ML models
    model = RandomForestClassifier(random_state=0)
    fuzzy_model = FuzzyClassifier(model, fuzzy_ctrl)

    # fit fuzzy model
    fuzzy_model.fit(X_visible, y_visible)
    score_fz = fuzzy_model.score(X, y)

    # fit standard model
    model.fit(X_visible, y_visible)
    score_dt = model.score(X, y)

    plt.figure(figsize=[10, 5])
    plt.subplot(121)
    plot_decision_function(fuzzy_model, "Fuzzy - acc %.2f" % score_fz, X, y, not_visible)
    plt.subplot(122)
    plot_decision_function(model, "No rules - acc %.2f" % score_dt, X, y, not_visible)
    plt.tight_layout()
    plt.savefig("./decision_boundaries.png")
    plt.show()

    return


if __name__ == "__main__":
    main()
