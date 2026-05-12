import numpy as np
import matplotlib.pyplot as plt
import warnings

warnings.simplefilter('ignore', np.RankWarning)

m = 100
np.random.seed(42)
X = 6 * np.random.rand(m, 1) - 4
y = 0.5 * X**2 + X + 2 + np.random.randn(m, 1)

X_flat = X.flatten()
y_flat = y.flatten()

np.random.seed(10)
indices = np.random.permutation(m)
train_size = int(0.8 * m)
X_train = X_flat[indices[:train_size]]
y_train = y_flat[indices[:train_size]]
X_val = X_flat[indices[train_size:]]
y_val = y_flat[indices[train_size:]]

def plot_learning_curves_custom(degree, ax, title):
    train_errors, val_errors = [], []
    for m_val in range(1, len(X_train) + 1):
        X_subset = X_train[:m_val]
        y_subset = y_train[:m_val]
        
        coefs = np.polyfit(X_subset, y_subset, degree)
        poly = np.poly1d(coefs)
        
        y_train_predict = poly(X_subset)
        y_val_predict = poly(X_val)
        
        train_mse = np.mean((y_subset - y_train_predict)**2)
        val_mse = np.mean((y_val - y_val_predict)**2)
        
        train_errors.append(train_mse)
        val_errors.append(val_mse)

    ax.plot(np.sqrt(train_errors), "r-+", linewidth=2, label="Навчальний набір")
    ax.plot(np.sqrt(val_errors), "b-", linewidth=3, label="Перевірочний набір")
    ax.set_xlabel("Розмір навчального набору")
    ax.set_ylabel("RMSE")
    ax.set_title(title)
    ax.legend()
    ax.set_ylim(0, 3)
    ax.grid(True, alpha=0.3)

fig, axes = plt.subplots(3, 1, figsize=(10, 15))

plot_learning_curves_custom(1, axes[0], "Рис. 12. Криві навчання для лінійної моделі")
plot_learning_curves_custom(10, axes[1], "Рис. 13. Криві навчання для поліноміальної моделі (ступінь 10)")
plot_learning_curves_custom(2, axes[2], "Рис. 14. Криві навчання для поліноміальної моделі (ступінь 2)")

plt.tight_layout()
plt.show()
