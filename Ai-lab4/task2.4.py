import urllib.request
import numpy as np
import matplotlib.pyplot as plt

url = "https://www4.stat.ncsu.edu/~boos/var.select/diabetes.tab.txt"
response = urllib.request.urlopen(url)
data = np.genfromtxt(response, skip_header=1, delimiter='\t')

X = data[:, :-1]
y = data[:, -1]

X_centered = X - X.mean(axis=0)
X = X_centered / np.sqrt(np.sum(X_centered**2, axis=0))

np.random.seed(0)
indices = np.arange(len(X))
np.random.shuffle(indices)

test_size = int(len(X) * 0.5)
train_idx = indices[test_size:]
test_idx = indices[:test_size]

Xtrain, ytrain = X[train_idx], y[train_idx]
Xtest, ytest = X[test_idx], y[test_idx]

Xtrain_lin = np.c_[np.ones(Xtrain.shape[0]), Xtrain]
Xtest_lin = np.c_[np.ones(Xtest.shape[0]), Xtest]

beta, _, _, _ = np.linalg.lstsq(Xtrain_lin, ytrain, rcond=None)

intercept = beta[0]
coefs = beta[1:]

ypred = Xtest_lin.dot(beta)

mae = np.mean(np.abs(ytest - ypred))
mse = np.mean((ytest - ypred)**2)
r2 = 1 - np.sum((ytest - ypred)**2) / np.sum((ytest - np.mean(ytest))**2)

print("Коефіцієнти регресії (coef_):")
print(coefs)
print("\nПеретин (intercept_):", intercept)
print("R2 score:", round(r2, 2))
print("Mean absolute error:", round(mae, 2))
print("Mean squared error:", round(mse, 2))

fig, ax = plt.subplots()
ax.scatter(ytest, ypred, edgecolors=(0, 0, 0))
ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=4)
ax.set_xlabel('Виміряно')
ax.set_ylabel('Передбачено')
plt.show()
