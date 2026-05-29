import numpy as np
from kernel2 import kernel_quintico
from geometry import calcular_distancias_periodicas

def _gradiente_sph_simetrico(field, grad_W, m, rho):

    """Gradiente 1D optimizado para rho y h constantes."""

    field_diff = field[:, np.newaxis] - field[np.newaxis, :]

    # Si rho es constante: 0.5 * m * (1/rho + 1/rho) = m / rho
    grad_factor = m / rho

    return np.sum(grad_factor * field_diff * grad_W, axis=1)

def _laplaciano_brookshaw(field, grad_W, dx, m, rho):

    """Laplaciano 1D optimizado para rho y h constantes."""

    field_diff = field[:, np.newaxis] - field[np.newaxis, :]

    # Si rho es constante: 4 * m / (2 * rho) = 2 * m / rho
    factor = 2.0 * m / rho

    psi = np.zeros_like(dx)
    mask = np.abs(dx) > 1e-12
    psi[mask] = grad_W[mask] / dx[mask]

    return np.sum(factor * field_diff * psi, axis=1)

def _hilbert_sph(field, dx, m, rho, L):

    """Transformada de Hilbert periódica optimizada."""

    mask = np.abs(dx) > 1e-12
    cot_kernel = np.zeros_like(dx)
    cot_kernel[mask] = 1.0 / np.tan(np.pi * dx[mask] / L)

    # Si rho es constante: m / rho es constante

    return (m / (rho * L)) * np.sum(field[np.newaxis, :] * cot_kernel, axis=1)

def evaluar_sistema_movil(x, u, m, rho, h, beta, Le, sigma, L):

    """
    Calcula las derivadas de la ecuación de Sivashinsky (1993) en 1D.
    Asume que rho y h son constantes para maximizar eficiencia.
    """

    dx = calcular_distancias_periodicas(x, L)

    # Con h constante, solo necesitamos una llamada al kernel
    _, grad_W = kernel_quintico(dx, h)

    # Coeficientes según Sivashinsky (1993):
    alpha = 0.5 * beta * (1.0 - Le) - 1.0
    gamma = 4.0
    lambda_param = 0.5 * (1.0 - sigma)

    # Operadores SPH:
    u_x = _gradiente_sph_simetrico(u, grad_W, m, rho)
    lap_u = _laplaciano_brookshaw(u, grad_W, dx, m, rho)
    
    u_xx = lap_u
    grad_lap_u = _gradiente_sph_simetrico(lap_u, grad_W, m, rho)
    u_xxx = -grad_lap_u
    u_xxxx = _laplaciano_brookshaw(lap_u, grad_W, dx, m, rho)

    # Términos de Hilbert:

    hilbert_u = _hilbert_sph(u, dx, m, rho, L)
    hilbert_ux = _hilbert_sph(u_x, dx, m, rho, L)
    
    I_h = (1.0 / (2.0 * np.pi)) * hilbert_u
    I_hx = (1.0 / (2.0 * np.pi)) * hilbert_ux

    # Derivadas materiales (marco lagrangiano):

    du_dt = -alpha * u_xx - gamma * u_xxxx + lambda_param * I_hx
    dh_dt = 0.5 * u**2 - alpha * u_x - gamma * u_xxx + lambda_param * I_h

    return du_dt, dh_dt, u
