import numpy as np
import matplotlib.pyplot as plt

X = np.array([0.1, 0.3, 0.4, 0.6, 0.7])
Y = np.array([3.2, 3.0, 1.0, 1.8, 1.9])

coefs = np.polyfit(X, Y, 4)
poly = np.poly1d(coefs)

X_interp = np.linspace(X.min(), X.max(), 100)
Y_interp = poly(X_interp)

plt.figure(figsize=(8, 5))
plt.scatter(X, Y, color='blue', s=50, label='Задані точки', zorder=5)
plt.plot(X_interp, Y_interp, color='red', label='Інтерполяційна крива')

plt.title('Інтерполяція функції (поліном 4-го степеня)')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.grid(True)
plt.show()
