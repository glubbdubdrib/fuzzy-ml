import copy

from scipy import stats
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import numpy as np
import scikit_posthocs as sp


def extract_rules(X, y, alpha=0.05, verbose=False):

    y_set = set(y)

    # consequent
    step = 0.1
    variance = 3
    label = ctrl.Consequent(np.arange(0, 10*(len(y_set)-1) + step, step), 'label')
    for yi in y_set:
        label[f'L{yi}'] = fuzz.gaussmf(label.universe, 10*yi, variance)
    if verbose:
        label.view()

    rule_dict = dict()
    extracted_rules = []
    for c in range(0, X.shape[1]):
        v = X.iloc[:, c].var() / 10
        val_range = np.arange(X.iloc[:, c].min() - v, X.iloc[:, c].max() + v, v)
        feature = ctrl.Antecedent(val_range, X.columns[c])
        y_set_2 = copy.deepcopy(y_set)
        for yi in y_set:
            y_set_2.remove(yi)
            for yj in y_set_2:
                x0, x1 = X.iloc[y==yi, c], X.iloc[y==yj, c]
                t, p = stats.ttest_ind(x0, x1, equal_var=False)
                m0, m1 = np.mean(x0), np.mean(x1)
                if p < alpha:
                    # rule A
                    rule_str = f'{X.columns[c]}[Y{yi}] -> L{yi}'
                    if rule_str not in rule_dict:
                        feature[f'Y{yi}'] = fuzz.gaussmf(feature.universe, m0, v)
                        ruleA = ctrl.Rule(feature[f'Y{yi}'], label[f'L{yi}'])
                        extracted_rules.append(ruleA)
                        rule_dict[rule_str] = True

                    # rule B
                    rule_str = f'{X.columns[c]}[Y{yj}] -> L{yj}'
                    if rule_str not in rule_dict:
                        feature[f'Y{yj}'] = fuzz.gaussmf(feature.universe, m1, v)
                        ruleB = ctrl.Rule(feature[f'Y{yj}'], label[f'L{yj}'])
                        extracted_rules.append(ruleB)
                        rule_dict[rule_str] = True

                    if verbose:
                        print(f"F{c}\n\t{m1:.2f} -> y=1 [p={p:.2f}]\n\t{m0:.2f} -> y=0")
                        feature.view()

    return extracted_rules
