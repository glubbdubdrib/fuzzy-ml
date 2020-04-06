import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import accuracy_score

from ._chull import convex_combination_test, convex_PCA


class FuzzyClassifier(BaseEstimator, ClassifierMixin):

    def __init__(self, model, fuzzy_ctrl, alpha=0.8, chull="PCA"):
        self.model = model
        self.fuzzy_ctrl = fuzzy_ctrl
        self.alpha = alpha
        self.chull = chull

    def fit(self, X, y):
        # Check that X and y have correct shape
        X, y = check_X_y(X, y)
        # Store the classes seen during fit
        self.classes_ = unique_labels(y)
        self.X_ = X

        self.model.fit(X, y)

        # Return the classifier
        return self

    def predict_proba(self, X):
        # Check is fit had been called
        check_is_fitted(self)

        # Input validation
        X_checked = check_array(X)

        y_fuzzy = []
        ctrl_inputs = [i.label for i in self.fuzzy_ctrl.ctrl.antecedents]
        consequent = [i for i in self.fuzzy_ctrl.ctrl.consequents][0]
        normalize = max(consequent.universe) - min(consequent.universe)
        for i in range(X.shape[0]):
            for ctrl_in in ctrl_inputs:
                self.fuzzy_ctrl.input[ctrl_in] = X.loc[i, ctrl_in]
            self.fuzzy_ctrl.compute()
            y_fuzzy.append(self.fuzzy_ctrl.output["label"] / normalize)

        y_fuzzy = np.array(y_fuzzy)
        y_fuzzy = np.vstack([y_fuzzy, 1-y_fuzzy]).T

        # compute convex hull
        if self.chull == "exact":
            is_out = convex_combination_test(self.X_, X)
        elif self.chull == "PCA":
            is_out = convex_PCA(self.X_, X)
        else:
            raise AttributeError

        X_out, X_in = X.iloc[is_out], X.iloc[~is_out]

        y = np.zeros((len(X), 2))
        if not X_in.empty:
            y_in = self.alpha * self.model.predict_proba(X_in) + (1-self.alpha) * y_fuzzy[~is_out]
            y[~is_out] = y_in
        if not X_out.empty:
            y_out = (1-self.alpha) * self.model.predict_proba(X_out) + self.alpha * y_fuzzy[is_out]
            y[is_out] = y_out

        return y

    def predict(self, X):
        y = self.predict_proba(X)
        return y[:, 1] >= 0.5

    def score(self, X, y, sample_weight=None):
        y_pred = self.predict(X)
        return accuracy_score(y, y_pred)
