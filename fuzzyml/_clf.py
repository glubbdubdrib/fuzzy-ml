import copy

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import IsolationForest
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import accuracy_score
import skfuzzy as fuzz

from ._chull import convex_combination_test, convex_PCA


class FuzzyClassifier(BaseEstimator, ClassifierMixin):

    def __init__(self, model, fuzzy_ctrl, alpha=0.8, novelty="detection",
                 detection=IsolationForest(), random_state=None):
        self.model = model
        self.fuzzy_ctrl = fuzzy_ctrl
        self.alpha = alpha
        self.novelty = novelty
        self.detection = detection
        self.random_state = random_state

    def fit(self, X, y):
        # Check that X and y have correct shape
        X, y = check_X_y(X, y)
        # Store the classes seen during fit
        self.classes_ = unique_labels(y)
        self.X_ = X
        self.y_set_ = set(y)
        self.detection_ = copy.deepcopy(self.detection)
        if self.random_state and hasattr(self.detection, "random_state"):
            setattr(self.detection_, "random_state", self.random_state)

        if self.novelty == "detection":
            self.detection.fit(X)

        self.model.fit(X, y)

        # Return the classifier
        return self

    def predict_proba(self, X):
        # Check is fit had been called
        check_is_fitted(self)

        # Input validation
        X_checked = check_array(X)

        ctrl_inputs = [i.label for i in self.fuzzy_ctrl.ctrl.antecedents]
        consequent = [i for i in self.fuzzy_ctrl.ctrl.consequents][0]
        y_fuzzy = np.zeros((X.shape[0], len(self.y_set_)))
        for nr, sample in enumerate(X.index):
            for ctrl_in in ctrl_inputs:
                self.fuzzy_ctrl.input[ctrl_in] = X.loc[sample, ctrl_in]
            self.fuzzy_ctrl.compute()
            label_pred = self.fuzzy_ctrl.output["label"]
            for nc, (_, term) in enumerate(consequent.terms.items()):
                y_fuzzy[nr, nc] = fuzz.interp_membership(consequent.universe,
                                                         term.mf,
                                                         label_pred)

        # compute convex hull
        if self.novelty == "CH":
            is_out = convex_combination_test(self.X_, X)
        elif self.novelty == "PCA":
            is_out = convex_PCA(self.X_, X)
        elif self.novelty == "detection":
            is_in = self.detection.predict(X)
            is_out = copy.deepcopy(is_in)
            is_out[is_in == 1] = False
            is_out[is_in == -1] = True
            is_out = is_out.astype(bool)
        else:
            raise AttributeError

        X_out, X_in = X.iloc[is_out], X.iloc[~is_out]

        y = np.zeros((len(X), len(self.y_set_)))
        if not X_in.empty:
            y_in = self.alpha * self.model.predict_proba(X_in) + (1 - self.alpha) * y_fuzzy[~is_out]
            y[~is_out] = y_in
        if not X_out.empty:
            y_out = (1 - self.alpha) * self.model.predict_proba(X_out) + self.alpha * y_fuzzy[is_out]
            y[is_out] = y_out

        return y

    def predict(self, X):
        y = self.predict_proba(X)
        return np.argmax(y, axis=1)

    def score(self, X, y, sample_weight=None):
        y_pred = self.predict(X)
        return accuracy_score(y, y_pred)
