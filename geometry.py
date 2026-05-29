import numpy as np
from kernel2 import kernel_quintico

def calcular_distancias_periodicas(x, L):
    """Calcula la matriz de distancias periódicas dx[i,j] = x[i] - x[j]."""
    dx = x[:, np.newaxis] - x[np.newaxis, :]
    return dx - L * np.round(dx / L)

def ajustar_rho_y_h(x, m, h, eta, L):
    """Calcula la densidad rho usando un h (longitud de suavizado) constante."""
    dx = calcular_distancias_periodicas(x, L)
    W, _ = kernel_quintico(dx, h[:, np.newaxis])
    rho = np.sum(m * W, axis=1)
    rho = np.maximum(rho, 1e-10)
    return rho, h
