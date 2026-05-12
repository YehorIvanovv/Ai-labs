import numpy as np
import matplotlib.pyplot as plt

X = np.array([7, 12, 17, 22, 27, 32])
Y = np.array([8, 7, 6, 5, 4, 3])

beta_1, beta_0 = np.polyfit(X, Y, 1)

print(f"Рівняння лінійної регресії: y = {beta_0:.2f} {beta_1:+.2f}x")

plt.figure(figsize=(8, 5))
plt.scatter(X, Y, color='blue', label='Експериментальні точки (X, Y)')
plt.plot(X, beta_0 + beta_1 * X, color='red', label=f'Апроксимуюча пряма (y={beta_0:.2f}{beta_1:+.2f}x)')
plt.title('Лінійна регресія. Метод найменших квадратів')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.grid(True)
plt.show()
