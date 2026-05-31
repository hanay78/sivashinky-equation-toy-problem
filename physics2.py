import numpy as np
from kernel2 import kernel_quintico
from geometry import calcular_distancias_periodicas

def _gradiente_sph_simetrico(field, grad_W, m, rho):
    """Gradiente 1D optimizado (field_j - field_i)."""
    field_diff = field[np.newaxis, :] - field[:, np.newaxis]
    vol_j = m / rho
    return np.sum(vol_j[np.newaxis, :] * field_diff * grad_W, axis=1)

def _laplaciano_brookshaw(field, grad_W, dx, m, rho):
    """Laplaciano 1D de Brookshaw (field_i - field_j)."""
    field_diff = field[:, np.newaxis] - field[np.newaxis, :]
    vol_j = m / rho
    psi = np.zeros_like(dx)
    mask = np.abs(dx) > 1e-12
    psi[mask] = grad_W[mask] / dx[mask]
    return np.sum(2.0 * vol_j[np.newaxis, :] * field_diff * psi, axis=1)

def _hilbert_sph(field, dx, m, rho, L):
    """Transformada de Hilbert periódica estable."""
    mask = np.abs(dx) > 1e-12
    cot_kernel = np.zeros_like(dx)
    cot_kernel[mask] = 1.0 / np.tan(np.pi * dx[mask] / L)
    vol_j = m / rho
    field_diff = field[np.newaxis, :] - field[:, np.newaxis]
    return np.sum((vol_j[np.newaxis, :] / L) * field_diff * cot_kernel, axis=1)

def evaluar_sistema_movil(x, u, h_field, m, rho, h_smooth, beta, Le, sigma, L):
    """
    Motor SPH para la ecuación de Sivashinsky (1993).
    Reescrito para basarse en el campo h y sus derivadas (1ra, 2da, 4ta).
    """
    dx = calcular_distancias_periodicas(x, L)

    # Con h_smooth constante, solo necesitamos una llamada al kernel
    _, grad_W = kernel_quintico(dx, h_smooth)

    # Coeficientes según Sivashinsky (1993):
    alpha = 0.5 * beta * (1.0 - Le) - 1.0
    gamma = 4.0
    lambda_param = 0.5 * (1.0 - sigma)

    # 1. Derivadas de h (usando gradiente y laplaciano Brookshaw)
    # h_x -> grad(h)
    # h_xx -> lap(h)
    # h_xxxx -> lap(lap(h))
    h_x = _gradiente_sph_simetrico(h_field, grad_W, m, rho)
    h_xx = _laplaciano_brookshaw(h_field, grad_W, dx, m, rho)
    h_xxxx = _laplaciano_brookshaw(h_xx, grad_W, dx, m, rho)

    # 2. Término de Hilbert para h: I(h) = H(h_x)
    # Este término es equivalente a |k|*h_hat en el espacio de Fourier.
    I_h = _hilbert_sph(h_x, dx, m, rho, L)

    # 3. Evolución de h (Marco Lagrangiano)
    # h_t_euler = -0.5*(h_x^2) - alpha*h_xx - gamma*h_xxxx + lambda*I_h
    # dh/dt = h_t_euler + u*h_x = 0.5*(h_x^2) - alpha*h_xx - gamma*h_xxxx + lambda*I_h
    dh_dt = 0.5 * (h_x**2) - alpha * h_xx - gamma * h_xxxx + lambda_param * I_h

    # 4. Evolución de u (u = h_x)
    # Obtenemos du/dt derivando la ecuación euleriana de h (h_t_euler = dh_dt - h_x^2)
    # du/dt_lagrange = (h_t_euler)_x + u * u_x = (h_t_euler)_x + h_x * h_xx
    h_t_euler = dh_dt - (h_x**2)
    du_dt = _gradiente_sph_simetrico(h_t_euler, grad_W, m, rho) + h_x * h_xx

    # 5. Velocidad de la malla (las partículas se mueven con la pendiente local h_x)
    v_malla = h_x

    return du_dt, dh_dt, v_malla
