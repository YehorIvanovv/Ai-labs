import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

m = 100
X = 6 * np.random.rand(m, 1) - 4
y = 0.5 * X ** 2 + X + 2 + np.random.randn(m, 1)

X_flat = X.flatten()
y_flat = y.flatten()

beta_lin = np.polyfit(X_flat, y_flat, 1)
poly_lin = np.poly1d(beta_lin)

beta_poly = np.polyfit(X_flat, y_flat, 2)
poly_quad = np.poly1d(beta_poly)

X_plot = np.linspace(X_flat.min(), X_flat.max(), 100)
y_plot_lin = poly_lin(X_plot)
y_plot_poly = poly_quad(X_plot)

y_pred_lin = poly_lin(X_flat)
y_pred_poly = poly_quad(X_flat)

mae_lin = np.mean(np.abs(y_flat - y_pred_lin))
mse_lin = np.mean((y_flat - y_pred_lin)**2)
r2_lin = 1 - np.sum((y_flat - y_pred_lin)**2) / np.sum((y_flat - np.mean(y_flat))**2)

mae_poly = np.mean(np.abs(y_flat - y_pred_poly))
mse_poly = np.mean((y_flat - y_pred_poly)**2)
r2_poly = 1 - np.sum((y_flat - y_pred_poly)**2) / np.sum((y_flat - np.mean(y_flat))**2)

print("Linear Regression Metrics:")
print(f"MAE: {mae_lin:.2f}")
print(f"MSE: {mse_lin:.2f}")
print(f"R2: {r2_lin:.2f}\n")

print("Polynomial Regression (Degree 2) Metrics:")
print(f"MAE: {mae_poly:.2f}")
print(f"MSE: {mse_poly:.2f}")
print(f"R2: {r2_poly:.2f}")

plt.figure(figsize=(10, 6))
plt.scatter(X_flat, y_flat, color='blue', label='Випадкові дані (Варіант 3)', alpha=0.6)
plt.plot(X_plot, y_plot_lin, color='red', linewidth=2, label='Лінійна регресія')
plt.plot(X_plot, y_plot_poly, color='green', linewidth=3, label='Поліноміальна регресія (ступінь 2)')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Лінійна та поліноміальна регресія')
plt.legend()
plt.grid(True)
plt.show()
