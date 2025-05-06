import numpy as np
from sklearn.svm import SVC
from sklearn.metrics.pairwise import rbf_kernel
import pandas as pd
import matplotlib.pyplot as plt
from cvxopt import matrix, solvers
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, GridSearchCV, KFold

class Polynomial():

    def __init__(self, M=1):
        self.M = M

    def __call__(self, A, B):
        return (1 + np.dot(A, B.T))**self.M
    
class RBF():

    def __init__(self, sigma=0.2):
        self.sigma = sigma

    def __call__(self, A, B):
        A = np.atleast_2d(A)
        B = np.atleast_2d(B)
        a = np.sum(A*A, axis=-1).reshape(-1, 1)
        b = np.sum(B*B, axis=-1).reshape(1, -1)
        dist = a - 2*np.dot(A, B.T) + b
        K = (np.exp(-(dist / (2*(self.sigma**2)))))
        return K.squeeze() if A.shape[0] == 1 or B.shape[0] == 1 else K

class KernelizedRidgeRegression():

    def __init__(self, kernel, lambda_=0.0):
        self.kernel = kernel
        self.lambda_ = lambda_
        self.alpha = None
        self.X_train = None

    def fit(self, X, y):
        self.X_train = X
        K = self.kernel(X, X)
        n = K.shape[0]
        self.alpha = np.linalg.solve(K + self.lambda_*np.eye(n), y)
        return self

    def predict(self, X):
        return np.dot(self.kernel(X, self.X_train), self.alpha)
    
class SVR():

    def __init__(self, kernel, lambda_=0.001, epsilon=0.1):
        self.kernel = kernel
        self.lambda_ = lambda_
        self.C = 1/self.lambda_
        self.epsilon = epsilon
        self.alphas = None
        self.alphas_star = None
        self.b = None
        self.X_train = None
        self.X_support = None
        self.y_support = None

    def fit(self, X, y):

        self.X_train = X
        n = X.shape[0]
        K = self.kernel(X, X)

        P = np.zeros((2 * n, 2 * n))
        for i in range(n):
            for j in range(n):
                P[2*i, 2*j] = K[i, j]
                P[2*i+1, 2*j+1] = K[i, j]
                P[2*i, 2*j+1] = -K[i, j]
                P[2*i+1, 2*j] = -K[i, j]

        #alphas should be [alpha_i, alpha_i*, ...] should be 2n of them right?
        A = np.zeros((1, 2*n))
        for i in range(n):
            A[0, 2*i] = 1   # α_i
            A[0, 2*i+1] = -1  # -α_i*
        A = matrix(A)
        b = matrix([0.0])

        G = matrix(np.vstack([
            -np.eye(2*n),
            np.eye(2*n)
        ]))
        h = matrix(np.hstack([
            np.zeros(2*n),
            self.C * np.ones(2*n)
        ]))

        q = np.zeros(2 * n)
        for i in range(n):
            q[2*i]   = self.epsilon - y[i]
            q[2*i+1] = self.epsilon + y[i]

        sol = solvers.qp(matrix(P), matrix(q), matrix(G), matrix(h), matrix(A), matrix(b), options={'show_progress': False})

        all_alphas = np.array(sol["x"]).flatten()
        self.alphas = all_alphas[::2]
        self.alphas_star = all_alphas[1::2]
        self.b = float(sol["y"][0])

        tol = 1e-5
        support_mask = (
            (self.alphas > tol) & (self.alphas < self.C - tol)
        ) | (
            (self.alphas_star > tol) & (self.alphas_star < self.C - tol)
        )
        support_indices = np.where(support_mask)[0]

        self.X_support = X[support_indices]
        self.y_support = y[support_indices]

        return self
    
    def predict(self, X):
        cs = self.alphas - self.alphas_star
        cs = cs.reshape(-1, 1)
        K = self.kernel(X, self.X_train)
        res = np.dot(K, cs) + self.b
        return res.flatten()
    
    def get_alpha(self):
        return np.stack([self.alphas, self.alphas_star], axis=1)

    def get_b(self):
        return self.b

class KRR_scikit(BaseEstimator, ClassifierMixin):

    def __init__(self, kernel, lambda_=0.0):
        self.kernel = kernel
        self.lambda_ = lambda_

    def fit(self, X, y):
        self.model = KernelizedRidgeRegression(self.kernel, self.lambda_)
        self.model.fit(X, y)
        return self
    
    def predict(self, X):
        return self.model.predict(X)

def sine_data():
    sine_df = pd.read_csv("sine.csv")
    X = sine_df["x"].to_numpy()
    y = sine_df["y"].to_numpy()

    X = X.reshape(-1, 1)
    y = y.reshape(-1)
    return X, y

def housing_data():
    housing_df = pd.read_csv("housing2r.csv")
    y = housing_df["y"].to_numpy()
    X = housing_df.drop(columns="y").to_numpy()
    return X, y

def plot_KRR_RBF(X, y, sigma=1, lambda_=0.1, save=False, ax=None):
    predictor = KernelizedRidgeRegression(RBF(sigma=sigma), lambda_=lambda_)
    predictor = predictor.fit(X, y)

    span = np.linspace(np.min(X), np.max(X), 200).reshape(-1, 1)
    predictions = predictor.predict(span)

    ax.scatter(X, y)
    ax.plot(span, predictions, color="r")

    if save:
        plt.savefig("kernelized_ridge_regression_RBF_sine.pdf", bbox_inches="tight")
    return ax

def plot_KRR_POLY(X, y, M=1, lambda_=0.1, save=False, ax=None):
    predictor = KernelizedRidgeRegression(Polynomial(M=M), lambda_=lambda_)
    predictor = predictor.fit(X, y)

    span = np.linspace(np.min(X), np.max(X), 200).reshape(-1, 1)
    predictions = predictor.predict(span)

    ax.scatter(X, y)
    ax.plot(span, predictions, color="r")

    if save:
        plt.savefig("kernelized_ridge_regression_POLY_sine.pdf", bbox_inches="tight")
    return ax

def plot_SVR_RBF(X, y, sigma=1, lambda_=0.1, save=False, ax=None):
    predictor = SVR(RBF(sigma=sigma), lambda_=lambda_)
    predictor = predictor.fit(X, y)

    span = np.linspace(np.min(X), np.max(X), 200).reshape(-1, 1)
    predictions = predictor.predict(span)

    ax.scatter(X, y)
    ax.plot(span, predictions, color="r")
    ax.scatter(predictor.X_support, predictor.y_support, edgecolor="black")

    if save:
        plt.savefig("support_vector_regression_RBF_sine.pdf", bbox_inches="tight")
    
    return ax

def plot_SVR_POLY(X, y, M=1, lambda_=0.1, save=False, ax=None):
    predictor = SVR(Polynomial(M=M), lambda_=lambda_, epsilon=0.1)
    predictor = predictor.fit(X, y)

    span = np.linspace(np.min(X), np.max(X), 200).reshape(-1, 1)
    predictions = predictor.predict(span)

    ax.scatter(X, y)
    ax.plot(span, predictions, color="r")
    ax.scatter(predictor.X_support, predictor.y_support, edgecolor="black")
    
    if save:
        plt.savefig("support_vector_regression_POLY_sine.pdf", bbox_inches="tight")

def CV_polynomial(model_class, ax, lambda_, support_vectors=False):
        s = []
        average_svs = []
        for i in range(1, 11):
            kf = KFold(n_splits=5, shuffle=True, random_state=42) #!return num of splits to 10
            cv_scores = []
            sv = []

            for train_indices, test_indices in kf.split(X):

                X_train, X_test = X[train_indices], X[test_indices]
                y_train, y_test = y[train_indices], y[test_indices]

                scaler = StandardScaler()
                scaler.fit(X_train)

                X_train = scaler.transform(X_train)
                X_test = scaler.transform(X_test)

                model = model_class(kernel=Polynomial(M=i), lambda_=lambda_)
                model = model.fit(X_train, y_train)

                predictions = model.predict(X_test)

                if support_vectors:
                    sv.append(len(model.X_support))

                cv_scores.append(np.mean((predictions - y_test)**2))

            s.append(np.mean(cv_scores))
            

            if support_vectors:
                average_svs.append(np.mean(sv))
                ax.text(i-1, s[-1], f"{average_svs[-1]:.0f}", fontsize=9, ha='center', va='bottom')

        ax.plot(range(1, len(s)+1), s, label="λ = 1")
        ax.grid(True)
        ax.set_xlabel("Degree")
        ax.set_ylabel("MSE")
        ax.legend()
        ax.set_yscale("log")

def CV_RBF(model_class, ax, lambda_, support_vectors=False):
    s = []
    average_svs = []
    sigmas = [0.001, 0.01, 0.1, 1, 2, 3, 4, 6, 10, 100]

    for i, sigma in enumerate(sigmas):
        kf = KFold(n_splits=5, shuffle=True, random_state=42) #!return num of splits to 10
        cv_scores = []
        sv = []
        
        for train_indices, test_indices in kf.split(X):

            X_train, X_test = X[train_indices], X[test_indices]
            y_train, y_test = y[train_indices], y[test_indices]

            scaler = StandardScaler()
            scaler.fit(X_train)

            X_train = scaler.transform(X_train)
            X_test = scaler.transform(X_test)

            model = model_class(kernel=RBF(sigma=sigma), lambda_=lambda_)
            model = model.fit(X_train, y_train)

            predictions = model.predict(X_test)

            if support_vectors:
                sv.append(len(model.X_support))

            cv_scores.append(np.mean((predictions - y_test)**2))

        s.append(np.mean(cv_scores))

        if support_vectors:
            average_svs.append(np.mean(sv))
            ax.text(i, s[-1], f"{average_svs[-1]:.0f}", fontsize=9, ha='center', va='bottom')

    ax.plot(range(len(s)), s, label="λ = 1")
    ax.grid(True)
    ax.set_xlabel("σ")
    ax.set_ylabel("MSE")
    ax.legend()

if __name__ == "__main__":
    # pol = RBF(sigma=0.2)
    # a = np.array([[1,2,3],
    #               [1,2,3]])
    # b = np.array([[1,1,1],
    #               [1,1,1]])
    
    # print(pol(a[0], b))

    # print(rbf_kernel(a, b, gamma=12.5))
    
    # X, y = sine_data()

    # fig, axes = plt.subplots(2, 2)
    # plot_KRR_POLY(X, y, M=5, lambda_=1, ax=axes[0][0])
    # plot_KRR_RBF(X, y, ax=axes[0][1])
    # plot_SVR_POLY(X, y, M=5, lambda_=1, ax=axes[1][0])
    # plot_SVR_RBF(X, y, ax=axes[1][1])
    # axes[0][0].set_title("KRR-Polynomial Kernel")
    # axes[0][1].set_title("KRR-RBF Kernel")
    # axes[1][0].set_title("SVR-Polynomial Kernel")
    # axes[1][1].set_title("SVR-RBF Kernel")
    # #TODO add legend and maybe pack this into function
    # plt.show()

    X, y = housing_data()
    fig, axes = plt.subplots(2, 2)

    CV_polynomial(KernelizedRidgeRegression, axes[0][0], lambda_=1)
    CV_polynomial(SVR, axes[0][1], lambda_=1, support_vectors=True)
    CV_RBF(KernelizedRidgeRegression, axes[1][0], lambda_=1)
    CV_RBF(SVR, axes[1][1], lambda_=1, support_vectors=True)
    plt.show()

    ##############################################################################
    #KFOLD USING WRAPPER
    ##############################################################################
        # s = []
    # for i in range(1, 11):

    #     m = Pipeline([
    #         ("scaler", StandardScaler()),
    #         ("Kernel Ridge Regression", KRR_scikit(kernel=Polynomial(M=i), lambda_=1))
    #     ])

    #     scores = cross_val_score(m, X, y, cv=KFold(n_splits=10, shuffle=True, random_state=42), scoring="neg_mean_squared_error")
    #     s.append(np.mean(np.abs(scores)))

    # axes[0].plot(range(len(s)), s)
    # axes[0].set_yscale("log")
    # plt.show()

    ###############################################################################
    #KFOLD WITHOUT WRAPPER
    ###############################################################################


    # m = Pipeline([
    #         ("scaler", StandardScaler()),
    #         ("Kernel Ridge Regression", KRR_scikit(kernel=Polynomial(M=1), lambda_=1))
    #     ])

    # param_grid = {"Kernel Ridge Regression__lambda_": [0.1, 0.5]}

    # inner_cv = GridSearchCV(m, param_grid, scoring="neg_mean_squared_error", cv=5)
    
    # outer_scores = cross_val_score(inner_cv, X, y, scoring="neg_mean_squared_error", cv=10)

    # print("Mean outer CV score:", -outer_scores.mean())





