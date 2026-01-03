# Locally Weighted Regression (LWR / LOWESS) - Non-Parametric Regression

import numpy as np
import matplotlib.pyplot as plt

# Generate synthetic dataset
np.random.seed(0)
X = np.linspace(0, 10, 50)
y = np.sin(X) + np.random.normal(0, 0.2, 50)

# Add bias term
X_mat = np.c_[np.ones(len(X)), X]

def locally_weighted_regression(x_query, X, y, tau):
    W = np.eye(len(X))
    for i in range(len(X)):
        diff = X[i][1] - x_query
        W[i, i] = np.exp(-(diff ** 2) / (2 * tau ** 2))

    theta = np.linalg.pinv(X.T @ W @ X) @ X.T @ W @ y
    return np.array([1, x_query]) @ theta

def predict(X, y, tau):
    y_pred = []
    for x in X[:, 1]:
        y_pred.append(locally_weighted_regression(x, X, y, tau))
    return np.array(y_pred)

# Bandwidth parameter
tau = 0.5

# Prediction
y_pred = predict(X_mat, y, tau)

# Plot
plt.figure()
plt.scatter(X, y, label="Data Points")
plt.plot(X, y_pred, label="LWR Fit")
plt.xlabel("X")
plt.ylabel("Y")
plt.legend()
plt.title("Locally Weighted Regression")
plt.show()
