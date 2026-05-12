import pickle
import numpy as np
import matplotlib.pyplot as plt

input_file = 'data_singlevar_regr.txt'
try:
    data = np.loadtxt(input_file, delimiter=',')
except ValueError:
    data = np.loadtxt(input_file)

X, y = data[:, :-1], data[:, -1]

num_training = int(0.8 * len(X))
X_train, y_train = X[:num_training], y[:num_training]
X_test, y_test = X[num_training:], y[num_training:]

X_train_flat = X_train.flatten()
X_test_flat = X_test.flatten()

beta_1, beta_0 = np.polyfit(X_train_flat, y_train, 1)
y_test_pred = beta_0 + beta_1 * X_test_flat

mae = np.mean(np.abs(y_test - y_test_pred))
mse = np.mean((y_test - y_test_pred)**2)
medae = np.median(np.abs(y_test - y_test_pred))
evs = 1 - np.var(y_test - y_test_pred) / np.var(y_test)
r2 = 1 - np.sum((y_test - y_test_pred)**2) / np.sum((y_test - np.mean(y_test))**2)

print("Linear regressor performance:")
print("Mean absolute error =", round(mae, 2))
print("Mean squared error =", round(mse, 2))
print("Median absolute error =", round(medae, 2))
print("Explain variance score =", round(evs, 2))
print("R2 score =", round(r2, 2))

output_model_file = 'model.pkl'
model_data = {'beta_0': beta_0, 'beta_1': beta_1}

with open(output_model_file, 'wb') as f:
    pickle.dump(model_data, f)

with open(output_model_file, 'rb') as f:
    regressor_model = pickle.load(f)

y_test_pred_new = regressor_model['beta_0'] + regressor_model['beta_1'] * X_test_flat
new_mae = np.mean(np.abs(y_test - y_test_pred_new))
print("\nNew mean absolute error =", round(new_mae, 2))

plt.scatter(X_test_flat, y_test, color='green')
plt.plot(X_test_flat, y_test_pred, color='black', linewidth=4)
plt.xticks(())
plt.yticks(())
plt.show()
