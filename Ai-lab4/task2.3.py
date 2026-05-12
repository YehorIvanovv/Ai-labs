import numpy as np
from itertools import combinations_with_replacement

try:
    data = np.loadtxt('data_multivar_regr.txt', delimiter=',')
except ValueError:
    data = np.loadtxt('data_multivar_regr.txt')

X, y = data[:, :-1], data[:, -1]

num_training = int(0.8 * len(X))
X_train, y_train = X[:num_training], y[:num_training]
X_test, y_test = X[num_training:], y[num_training:]

X_train_lin = np.c_[np.ones(X_train.shape[0]), X_train]
X_test_lin = np.c_[np.ones(X_test.shape[0]), X_test]

beta_lin, _, _, _ = np.linalg.lstsq(X_train_lin, y_train, rcond=None)

y_test_pred = X_test_lin.dot(beta_lin)

mae = np.mean(np.abs(y_test - y_test_pred))
mse = np.mean((y_test - y_test_pred)**2)
medae = np.median(np.abs(y_test - y_test_pred))
evs = 1 - np.var(y_test - y_test_pred) / np.var(y_test)
r2 = 1 - np.sum((y_test - y_test_pred)**2) / np.sum((y_test - np.mean(y_test))**2)

print("Linear Regressor performance:")
print("Mean absolute error =", round(mae, 2))
print("Mean squared error =", round(mse, 2))
print("Median absolute error =", round(medae, 2))
print("Explained variance score =", round(evs, 2))
print("R2 score =", round(r2, 2))

def get_poly_features(X_data, degree):
    n_samples, n_features = X_data.shape
    features = [np.ones(n_samples)]
    for d in range(1, degree + 1):
        for comb in combinations_with_replacement(range(n_features), d):
            features.append(np.prod(X_data[:, comb], axis=1))
    return np.column_stack(features)

X_train_poly = get_poly_features(X_train, 10)
beta_poly, _, _, _ = np.linalg.lstsq(X_train_poly, y_train, rcond=None)

datapoint = np.array([[7.75, 6.35, 5.56]])
datapoint_lin = np.c_[np.ones(1), datapoint]
datapoint_poly = get_poly_features(datapoint, 10)

print("\nLinear regression:\n", np.round(datapoint_lin.dot(beta_lin), 2))
print("\nPolynomial regression:\n", np.round(datapoint_poly.dot(beta_poly), 2))
