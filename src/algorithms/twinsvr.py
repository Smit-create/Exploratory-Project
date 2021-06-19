# Class of Twin Support Vector Regression

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics import r2_score
from cvxopt import matrix
from cvxopt import solvers


class TwinSVR(BaseEstimator, RegressorMixin):
    """Twin Supoort Vector Regression"""

    def __init__(
        self,
        c1=10,
        c2=10,
        e1=0.01,
        e2=0.01,
        kernel="gaussian_kernel",
        sigma=5,
        degree=3,
        regul=0,
    ):
        self.c1 = c1
        self.c2 = c2
        self.e1 = e1
        self.e2 = e2
        self.kernel = kernel
        self.sigma = sigma
        self.degree = degree
        self.regul = regul
        self.setup = False

    def fit(self, X, Y):
        """
        To fit the data and find the coefficients of the hyperplanes and the intercepts.

        Parameters
        ==========

        X : array/dataframe
            Contains the data features.
        Y : array
            Contains true labels of the data X.

        """
        self.setup = True
        m, n = X.shape
        self.X = X
        Y = np.array(Y).reshape(-1, 1)
        f = Y - self.e1
        h = Y + self.e2
        if self.kernel == "gaussian_kernel":
            G = matrix(
                np.hstack((self.gaussian_kernel(np.array(self.X)), np.ones((m, 1))))
            )

        elif self.kernel == "polynomial_kernel":
            G = matrix(np.hstack((self.polynomial_kernel(self.X), np.ones((m, 1)))))

        else:
            G = matrix(np.hstack((self.linear_kernel(self.X), np.ones((m, 1)))))
        [w1, b1] = self.plane1(G, f, m)
        [w2, b2] = self.plane1(G, h, m)
        self.coef_ = (w1 + w2) / 2
        self.intercept_ = (b1 + b2) / 2

    def predict(self, Xi):
        """
        To predict the results of test data using the hyperplane obtained from fitted data.

        Parameters
        ==========

        Xi : array/dataframe
            Contains the features of the data on which labels are to be predicted.

        Returns
        =======

        pre : array
            Contains the predicted labels of the data Xi.

        """
        if not self.setup:
            raise (Exception, "You must fit your data first.")
        if self.kernel == "gaussian_kernel":
            Xit = self.gaussian_kernel_predict(np.array(Xi))
        elif self.kernel == "polynomial_kernel":
            Xit = self.polynomial_kernel(Xi)
        else:
            Xit = self.linear_kernel(Xi)
        self.pre = np.dot(np.array(Xit), self.coef_).sum(axis=1) + self.intercept_
        return self.pre

    def score(self, Xt, yt):
        """
        To evaluate the performance of the results.

        Parameters
        ==========

        Xt: array/dataframe
            Contains the features of the data on which labels are to be predicted.
        yt: array
            Contains the true labels of the data Xt.

        Returns
        =======

        R2 score of predicted and true labels
        """
        ypre = self.predict(np.array(Xt))
        return r2_score(np.array(yt), ypre)

    def plane1(self, G, f, m):
        """To find the intercept and the coefficient of plane 1."""

        Z1 = np.dot(G.T, G)
        Z1 = Z1 + self.regul * (np.identity(Z1.shape[0]))
        Z2 = np.linalg.solve(Z1, G.T)
        Z3 = np.dot(G, Z2)
        P1 = matrix((Z3 + Z3.T) / 2)
        q1 = matrix(((f.T) - (np.dot(f.T, P1))).reshape(-1, 1))
        constraint1 = matrix(np.vstack((np.eye(m) * -1, np.eye(m))))
        constraint_ineq1 = matrix(np.hstack((np.zeros(m), np.ones(m) * self.c1)))
        alpha = solvers.qp(P=P1, q=q1, G=constraint1, h=constraint_ineq1)
        z = np.array(alpha["x"])
        w1 = np.dot(Z2, (f - z))[:-1]
        b1 = np.dot(Z2, (f - z))[-1]
        return [w1, b1]

    def plane2(self, G, h, m):
        """To find the intercept and the coeffcient of plane 2."""

        Z1 = np.dot(G.T, G)
        Z1 = Z1 + self.regul * (np.identity(Z1.shape[0]))
        Z2 = np.linalg.solve(Z1, G.T)
        Z3 = np.dot(G, Z2)
        P2 = matrix((Z3 + Z3.T) / 2)
        q2 = matrix(-1 * ((h.T) - (np.dot(h.T, P2))).reshape(-1, 1))
        constraint2 = matrix(np.vstack((np.eye(m) * -1, np.eye(m))))
        constraint_ineq2 = matrix(np.hstack((np.zeros(m), np.ones(m) * self.c2)))
        beta = solvers.qp(P=P2, q=q2, G=constraint2, h=constraint_ineq2)
        z = np.array(beta["x"])
        w2 = np.dot(Z2, (h + z))[:-1]
        b2 = np.dot(Z2, (h + z))[-1]
        return [w2, b2]

    # Definitions of various kernels
    def linear_kernel(self, x):
        return np.dot(x, (self.X).T)

    def polynomial_kernel(self, x):
        return (1 + np.dot(x, (self.X).T)) ** self.degree

    def gaussian_kernel(self, x):
        K = np.zeros((x.shape[0], x.shape[0]))
        for i in range(x.shape[0]):
            for j in range(x.shape[0]):
                K[i, j] = np.exp(
                    -(
                        np.linalg.norm(
                            np.array(x)[i].reshape(-1, 1)
                            - np.array(x)[j].reshape(-1, 1)
                        )
                        / (2 * (self.sigma ** 2))
                    )
                )
        return K

    def gaussian_kernel_predict(self, Xi):
        K = np.zeros((Xi.shape[0], (self.X).shape[0]))
        for i in range(Xi.shape[0]):
            for j in range((self.X).shape[0]):
                K[i, j] = np.exp(
                    -(
                        np.linalg.norm(
                            np.array(Xi)[i].reshape(-1, 1)
                            - np.array(self.X)[j].reshape(-1, 1)
                        )
                        / (2 * (self.sigma ** 2))
                    )
                )
        return K
