from scipy.optimize import linprog
from sklearn.decomposition import PCA
import numpy as np

def convex_PCA(X_trainval, X_test):
    pca = PCA()
    pca.fit(X_trainval)
    X_trainval_pca = pca.transform(X_trainval)
    X_test_pca = pca.transform(X_test)

    # find parallelepiped around training data and compute the volume
    lims = np.zeros((X_trainval_pca.shape[1], 2))
    for j in range(0, X_trainval_pca.shape[1]):
        lims[j, 0] = np.min(X_trainval_pca[:, j])
        lims[j, 1] = np.max(X_trainval_pca[:, j])
    # volume = np.prod( lims[:, 1]-lims[:, 0] )

    test_out = np.zeros(X_test_pca.shape[0])
    for j in range(0, X_test_pca.shape[1]):
        test_out = test_out + (X_test_pca[:, j] < lims[j, 0]).astype('int')
        test_out = test_out + (X_test_pca[:, j] > lims[j, 1]).astype('int')
    test_out = test_out > 0
    test_out = test_out.astype('bool')

    return test_out


def convex_combination_test(X_trainval, X_test):

    out_indeces = np.zeros(X_test.shape[0])
    for i in range(X_test.shape[0]):
        try:
            out_indeces[i] = not in_hull(X_trainval, X_test[i, :])
        except ValueError:
            out_indeces[i] = True
    out_indeces = out_indeces.astype('bool')

    return out_indeces


def in_hull(points, x):
    n_points = len(points)
    c = np.zeros(n_points)
    A = np.r_[points.T,np.ones((1,n_points))]
    b = np.r_[x, np.ones(1)]
    lp = linprog(c, A_eq=A, b_eq=b)
    return lp.success
