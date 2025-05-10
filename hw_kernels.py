import numpy as np
from sklearn.svm import SVC
from sklearn.metrics.pairwise import rbf_kernel
import pandas as pd
import matplotlib.pyplot as plt
from cvxopt import matrix, solvers
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, GridSearchCV, KFold, cross_val_predict

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

    def __init__(self, kernel, lambda_=0.001, epsilon=15):
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
        # support_mask = (
        #     (self.alphas > tol) & (self.alphas < self.C - tol)
        # ) | (
        #     (self.alphas_star > tol) & (self.alphas_star < self.C - tol)
        # )
        support_mask = self.alphas - self.alphas_star > tol
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

    def __init__(self, kernel, lambda_=0.1):
        self.kernel = kernel
        self.lambda_ = lambda_

    def fit(self, X, y):
        self.model = KernelizedRidgeRegression(self.kernel, self.lambda_)
        self.model.fit(X, y)
        return self
    
    def predict(self, X):
        return self.model.predict(X)

class SVR_scikit(BaseEstimator, ClassifierMixin):

    def __init__(self, kernel, lambda_=0.1):
        self.kernel = kernel
        self.lambda_ = lambda_

    def fit(self, X, y):
        self.model = SVR(self.kernel, self.lambda_)
        self.model.fit(X, y)
        support_vector_numbers.append(len(self.model.X_support))
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

    ax.scatter(X, y, label="Data")
    ax.plot(span, predictions, color="r", label="Predicted values")
    ax.legend()


    if save:
        plt.savefig("kernelized_ridge_regression_RBF_sine.pdf", bbox_inches="tight")
    return ax

def plot_KRR_POLY(X, y, M=1, lambda_=0.1, save=False, ax=None):
    predictor = KernelizedRidgeRegression(Polynomial(M=M), lambda_=lambda_)
    predictor = predictor.fit(X, y)

    span = np.linspace(np.min(X), np.max(X), 200).reshape(-1, 1)
    predictions = predictor.predict(span)

    ax.scatter(scaler.inverse_transform(X), y, label="Data")
    ax.plot(scaler.inverse_transform(span), predictions, color="r", label="Predicted values")
    ax.legend()

    if save:
        plt.savefig("kernelized_ridge_regression_POLY_sine.pdf", bbox_inches="tight")
    return ax

def plot_SVR_RBF(X, y, sigma=1, lambda_=0.1, epsilon=0.1, save=False, ax=None):
    predictor = SVR(RBF(sigma=sigma), lambda_=lambda_, epsilon=epsilon)
    predictor = predictor.fit(X, y)

    span = np.linspace(np.min(X), np.max(X), 200).reshape(-1, 1)
    predictions = predictor.predict(span)

    ax.scatter(X, y, label="Data")
    ax.plot(span, predictions, color="r", label="Predicted values")
    ax.scatter(predictor.X_support, predictor.y_support, edgecolor="black", label="Support vectors")
    ax.legend()

    if save:
        plt.savefig("support_vector_regression_RBF_sine.pdf", bbox_inches="tight")
    
    return ax

def plot_SVR_POLY(X, y, M=1, lambda_=0.1, epsilon=0.1, save=False, ax=None):
    predictor = SVR(Polynomial(M=M), lambda_=lambda_, epsilon=epsilon)
    predictor = predictor.fit(X, y)

    span = np.linspace(np.min(X), np.max(X), 200).reshape(-1, 1)
    predictions = predictor.predict(span)

    ax.scatter(scaler.inverse_transform(X), y, label="Data")
    ax.plot(scaler.inverse_transform(span), predictions, color="r", label="Predicted values")
    ax.scatter(scaler.inverse_transform(predictor.X_support), predictor.y_support, edgecolor="black", label="Support vectors")
    ax.legend()

    if save:
        plt.savefig("support_vector_regression_POLY_sine.pdf", bbox_inches="tight")

def CV_RBF(X, y, model_class, ax, lambda_, support_vectors=False):

    MSEs = []
    standard_errors = []
    mean_support_vectors = []
    sigmas = [0.001, 0.01, 0.1, 1, 2, 3, 4, 6, 10, 100]
    for i, sigma in enumerate(sigmas):

        global support_vector_numbers #i love poor coding practices.
        support_vector_numbers = []

        m = Pipeline([
            ("scaler", StandardScaler()),
            ("Kernel Ridge Regression", model_class(kernel=RBF(sigma=sigma), lambda_=lambda_))
        ])

        predictions = cross_val_predict(m, X, y, cv=KFold(n_splits=10, shuffle=True, random_state=42))
        
        errors = np.square(predictions - y)
        MSEs.append(np.mean(errors))
        if support_vectors:
            mean_support_vectors.append(np.mean(support_vector_numbers))
        standard_errors.append(np.std(errors)/np.sqrt(len(errors)))

    return MSEs, standard_errors, mean_support_vectors
   
def CV_polynomial(X, y, model_class, ax, lambda_, support_vectors=False):

    MSEs = []
    standard_errors = []
    mean_support_vectors = []
    for i in range(1, 11):

        global support_vector_numbers #i love poor coding practices.
        support_vector_numbers = []
        
        m = Pipeline([
            ("scaler", StandardScaler()),
            ("Kernel Ridge Regression", model_class(kernel=Polynomial(M=i), lambda_=lambda_))
        ])

        predictions = cross_val_predict(m, X, y, cv=KFold(n_splits=10, shuffle=True, random_state=42))
            
        errors = np.square(predictions - y)
        MSEs.append(np.mean(errors))
        if support_vectors:
            mean_support_vectors.append(np.mean(support_vector_numbers))
        standard_errors.append(np.std(errors)/np.sqrt(len(errors)))
    
    return MSEs, standard_errors, mean_support_vectors

def save_np_arrays(MSEs, errors, support_vectors, name):
    np.save(f"data/MSEs_{name}", MSEs)
    np.save(f"data/errors_{name}", errors)
    if support_vectors != []:
        np.save(f"data/support_vectors_{name}", support_vectors)

def read_np_arrays(name, sv=False):
    MSEs = np.load(f"data/MSEs_{name}.npy")
    errors = np.load(f"data/errors_{name}.npy")
    if sv:
        support_vectors  = np.load(f"data/support_vectors_{name}.npy")
        return MSEs, errors, support_vectors

    return MSEs, errors

def internal_CV(X, y, model, kernel):

    m = Pipeline([
                ("scaler", StandardScaler()),
                ("Regression", model(kernel=kernel, lambda_=0.1))
            ])

    inner_cv = KFold(n_splits=6, shuffle=True, random_state=42)
    outer_cv = KFold(n_splits=10, shuffle=True, random_state=42)

    global support_vector_numbers #i love poor coding practices.
    support_vector_numbers = []

    all_predictions = []
    all_y = []
    best_lambdas = []
    param_grid = {"Regression__lambda_": [0.0001, 0.001, 0.01, 0.1, 1, 10, 100]}

    for train_idx, test_idx in outer_cv.split(X):

            # Split the data for the outer fold
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            clf = GridSearchCV(estimator=m, param_grid=param_grid, cv=inner_cv, scoring="neg_mean_squared_error")
            clf.fit(X_train, y_train)

            best_lambda = clf.best_params_["Regression__lambda_"]
            best_lambdas.append(best_lambda)

            opt_model = Pipeline([
                ("scaler", StandardScaler()),
                ("Regression", model(kernel=kernel, lambda_=best_lambda))
            ])

            opt_model.fit(X_train, y_train)
            predictions = opt_model.predict(X_test)
            # predictions = cross_val_predict(opt_model, X=X, y=y, cv=outer_cv)
            all_predictions.append(predictions)
            all_y.append(y_test)

    return all_predictions, all_y, support_vector_numbers, best_lambdas

def CV_polynomial_internal(X, y, model_class, support_vectors=False):
    MSEs = []
    standard_errors = []
    mean_support_vectors = []
    all_best_lambdas = []

    for i in range(1, 11):

        all_predictions, all_y, support_vector_numbers, best_lambdas = internal_CV(X, y, model_class, kernel=Polynomial(M=i))
        
        errors = np.square(np.array(all_predictions).flatten() - np.array(all_y).flatten())
        MSEs.append(np.mean(errors))
        standard_errors.append(np.std(errors)/np.sqrt(len(errors)))
        all_best_lambdas.append(best_lambdas)

        if support_vectors:
            mean_support_vectors.append(np.mean(support_vector_numbers))

    return MSEs, standard_errors, mean_support_vectors, np.array(all_best_lambdas)

def CV_RBF_internal(X, y, model_class, support_vectors=False):
    MSEs = []
    standard_errors = []
    sigmas= [0.001, 0.01, 0.1, 1, 2, 3, 4, 6, 10, 100]
    mean_support_vectors = []
    all_best_lambdas = []

    for i, sigma in enumerate(sigmas):

        all_predictions, all_y, support_vector_numbers, best_lambdas = internal_CV(X, y, model_class, kernel=RBF(sigma=sigma))
        
        errors = np.square(np.array(all_predictions).flatten() - np.array(all_y).flatten())
        MSEs.append(np.mean(errors))
        standard_errors.append(np.std(errors)/np.sqrt(len(errors)))
        all_best_lambdas.append(best_lambdas)

        if support_vectors:
            mean_support_vectors.append(np.mean(support_vector_numbers))

    return MSEs, standard_errors, mean_support_vectors, np.array(all_best_lambdas)

if __name__ == "__main__":
    # pol = RBF(sigma=0.2)
    # a = np.array([[1,2,3],
    #               [1,2,3]])
    # b = np.array([[1,1,1],
    #               [1,1,1]])
    
    # print(pol(a[0], b))

    # print(rbf_kernel(a, b, gamma=12.5))
    
    # X, y = sine_data()
    
    # scaler = StandardScaler()
    # X_scaled = scaler.fit_transform(X)

    # fig, axes = plt.subplots(2, 1, figsize=(10, 8))
    # plot_KRR_POLY(X_scaled, y, M=10, lambda_=0.0001, ax=axes[0])
    # plot_KRR_RBF(X, y, ax=axes[1])
    # axes[0].set_title("KRR-Polynomial Kernel, deg = 10, λ = 0.0001")
    # axes[1].set_title("KRR-RBF Kernel, σ = 1, λ = 0.1")
    # # plt.show()
    # plt.savefig("sine1.pdf", bbox_inches="tight")

    # fig, axes = plt.subplots(2, 1, figsize=(10, 8))
    # plot_SVR_POLY(X_scaled, y, M=10, lambda_=0.0001, ax=axes[0], epsilon=0.5)
    # plot_SVR_RBF(X, y, sigma=1, lambda_=0.001, ax=axes[1], epsilon=0.5)
    # axes[0].set_title("SVR-Polynomial Kernel, deg= 10, λ = 0.0001, ε = 0.5")
    # axes[1].set_title("SVR-RBF Kernel, σ = 1, λ = 0.001, ε = 0.5")
    # plt.savefig("sine2.pdf", bbox_inches="tight")

    X, y = housing_data()
    fig, axes = plt.subplots(2, 1, sharey="col", sharex="col", figsize=(10, 8))


    # MSEs_KRR_polynomial, errors_KRR_polynomial, support_vectors_KRR_polynomial = CV_polynomial(X, y, KRR_scikit, axes[0][0], lambda_=1)
    # save_np_arrays(MSEs_KRR_polynomial, errors_KRR_polynomial, support_vectors_KRR_polynomial, "KRR_polynomial")
    # MSEs_SVR_polynomial, errors_SVR_polynomial, support_vectors_SVR_polynomial = CV_polynomial(X, y, SVR_scikit, axes[1], lambda_=1, support_vectors=True)
    # save_np_arrays(MSEs_SVR_polynomial, errors_SVR_polynomial, support_vectors_SVR_polynomial, "SVR_polynomial")
    # MSEs_KRR_RBF, errors_KRR_RBF, support_vectors_KRR_RBF = CV_RBF(X, y, KRR_scikit, axes[1][0], lambda_=1)
    # save_np_arrays(MSEs_KRR_RBF, errors_KRR_RBF, support_vectors_KRR_RBF, "KRR_RBF")
    # MSEs_SVR_RBF, errors_SVR_RBF, support_vectors_SVR_RBF = CV_RBF(X, y, SVR_scikit, axes[1][1], lambda_=1, support_vectors=True)
    # save_np_arrays(MSEs_SVR_RBF, errors_SVR_RBF, support_vectors_SVR_RBF, "SVR_RBF")

    # MSEs_KRR_polynomial_opt_lambda, errors_KRR_polynomial_opt_lambda, support_vectors_KRR_polynomial_opt_lambda, lambdas_KRR_polynomial_opt_lambda = CV_polynomial_internal(X, y , KRR_scikit)
    # save_np_arrays(MSEs_KRR_polynomial_opt_lambda, errors_KRR_polynomial_opt_lambda, support_vectors_KRR_polynomial_opt_lambda, "KRR_polynomial_opt_lambda")#
    # np.save("data/lambdas_KRR_polynomial_opt_lambda", lambdas_KRR_polynomial_opt_lambda)
    # MSEs_SVR_polynomial_opt_lambda, errors_SVR_polynomial_opt_lambda, support_vectors_SRV_polynomial_opt_lambda, lambdas_SVR_polynomial_opt_lambda = CV_polynomial_internal(X, y , SVR_scikit, True)
    # save_np_arrays(MSEs_SVR_polynomial_opt_lambda, errors_SVR_polynomial_opt_lambda, support_vectors_SRV_polynomial_opt_lambda, "SVR_polynomial_opt_lambda")
    # np.save("data/lambdas_SVR_polynomial_opt_lambda", lambdas_SVR_polynomial_opt_lambda)
    # MSEs_KRR_RBF_opt_lambda, errors_KRR_RBF_opt_lambda, support_vectors_KRR_RBF_opt_lambda, lambdas_KRR_RBF_opt_lambda = CV_RBF_internal(X, y, KRR_scikit)
    # save_np_arrays(MSEs_KRR_RBF_opt_lambda, errors_KRR_RBF_opt_lambda, support_vectors_KRR_RBF_opt_lambda, "KRR_RBF_opt_lambda")
    # np.save("data/lambdas_KRR_RBF_opt_lambda", lambdas_KRR_RBF_opt_lambda)
    # MSEs_SVR_RBF_opt_lambda, errors_SVR_RBF_opt_lambda, support_vectors_SVR_RBF_opt_lambda, lambdas_SVR_RBF_opt_lambda = CV_RBF_internal(X, y , SVR_scikit, True)
    # save_np_arrays(MSEs_SVR_RBF_opt_lambda, errors_SVR_RBF_opt_lambda, support_vectors_SVR_RBF_opt_lambda, "SVR_RBF_opt_lambda")
    # np.save("data/lambdas_SVR_RBF_opt_lambda", lambdas_SVR_RBF_opt_lambda)



    MSEs_KRR_polynomial, errors_KRR_polynomial = read_np_arrays("KRR_polynomial")
    MSEs_SVR_polynomial, errors_SVR_polynomial, support_vectors_SVR_polynomial = read_np_arrays("SVR_polynomial", True)
    MSEs_KRR_RBF, errors_KRR_RBF = read_np_arrays("KRR_RBF")
    MSEs_SVR_RBF, errors_SVR_RBF, support_vectors_SVR_RBF = read_np_arrays("SVR_RBF", True)

    
    MSEs_KRR_polynomial_opt_lambda, errors_KRR_polynomial_opt_lambda = read_np_arrays("KRR_polynomial_opt_lambda")
    MSEs_SVR_polynomial_opt_lambda, errors_SVR_polynomial_opt_lambda, support_vectors_SVR_polynomial_opt_lambda = read_np_arrays("SVR_polynomial_opt_lambda", True)
    MSEs_KRR_RBF_opt_lambda, errors_KRR_RBF_opt_lambda = read_np_arrays("KRR_RBF_opt_lambda")
    MSEs_SVR_RBF_opt_lambda, errors_SVR_RBF_opt_lambda, support_vectors_SVR_RBF_opt_lambda = read_np_arrays("SVR_RBF_opt_lambda", True)



    axes[0].errorbar(range(1, len(MSEs_KRR_polynomial)+1), MSEs_KRR_polynomial, yerr=errors_KRR_polynomial, linestyle="", marker=0, capsize=2, markersize="10", label="λ = 1")
    axes[1].errorbar(range(1, len(MSEs_SVR_polynomial)+1), MSEs_SVR_polynomial, yerr=errors_SVR_polynomial, linestyle="", marker=0, capsize=2, markersize="10", label="λ = 1")

    axes[0].errorbar(range(1, len(MSEs_KRR_polynomial_opt_lambda) + 1), MSEs_KRR_polynomial_opt_lambda, color="red", yerr=errors_KRR_polynomial_opt_lambda, linestyle="", marker=1, capsize=2, markersize="10", label="Optimal λ")
    axes[1].errorbar(range(1, len(MSEs_SVR_polynomial_opt_lambda) + 1), MSEs_SVR_polynomial_opt_lambda, color="red", yerr=errors_SVR_polynomial_opt_lambda, linestyle="", marker=1, capsize=2, markersize="10", label="Optimal λ")
    axes[0].set_yscale("log")

    for i in range(1, 11):
        axes[1].text(i-0.3, MSEs_SVR_polynomial[i-1] + 20, f"{support_vectors_SVR_polynomial[i-1]}", color="blue")

    for j in range(1, 11):
        axes[1].text(j+0.1, MSEs_SVR_polynomial_opt_lambda[j-1] - 45, f"{support_vectors_SVR_polynomial_opt_lambda[j-1]:.1f}", color="red")

    axes[0].grid(True)
    axes[1].grid(True)
    axes[0].set_xlabel("Polynomial Degree")
    axes[1].set_xlabel("Polynomial Degree")
    axes[0].set_ylabel("MSE")
    axes[1].set_ylabel("MSE")
    axes[0].set_title("Kernel Ridge Regression")
    axes[1].set_title("Support Vector Regression")

    axes[1].set_xticks(range(1,11))

    axes[0].legend()
    axes[1].legend()
    
    plt.show()

    
    # axes[1][0].errorbar(range(len(MSEs_KRR_RBF)), MSEs_KRR_RBF, yerr=errors_KRR_RBF, linestyle="", marker="o", markersize="3", label="λ = 1")
    # axes[1][1].errorbar(range(len(MSEs_SVR_RBF)), MSEs_SVR_RBF, yerr=errors_SVR_RBF, linestyle="", marker="o", markersize="3", label="λ = 1")


    # axes[0][1].set_yscale("log")



    
    # axes[1][0].errorbar(range(len(MSEs_KRR_RBF_opt_lambda)), MSEs_KRR_RBF_opt_lambda, color="red", yerr=errors_KRR_RBF_opt_lambda, linestyle="", marker="o", markersize="3", label="Optimal λ")
    # axes[1][1].errorbar(range(len(MSEs_SVR_RBF_opt_lambda)), MSEs_SVR_RBF_opt_lambda, color="red", yerr=errors_SVR_RBF_opt_lambda, linestyle="", marker="o", markersize="3", label="Optimal λ")
    

    # axes[1][0].grid(True)
    # axes[1][1].grid(True)

    # axes[1][0].legend()
    # axes[1][1].legend()
    


    ##############################################################################
    #KFOLD USING WRAPPER
    ##############################################################################
    # s = []
    # standard_errors = []
    # mean_support_vectors = []
    # for i in range(1, 11):

    #     support_vector_numbers = []
    #     m = Pipeline([
    #         ("scaler", StandardScaler()),
    #         ("Kernel Ridge Regression", SVR_scikit(kernel=Polynomial(M=i), lambda_=1))
    #     ])

    #     predictions = cross_val_predict(m, X, y, cv=KFold(n_splits=2, shuffle=True, random_state=42))
        
    #     errors = np.square(predictions - y)
    #     s.append(np.mean(errors))
    #     mean_support_vectors.append(np.mean(support_vector_numbers))
    #     standard_errors.append(np.std(errors)/np.sqrt(len(errors)))
    
    # plt.errorbar(range(len(s)), s, yerr=standard_errors, linestyle="", marker="o", markersize="3")
    # plt.yscale("log")

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





