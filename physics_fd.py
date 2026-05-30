import numpy as np

def evaluar_sistema_movil(x, u, h_field, m, rho, h_smooth, beta, Le, sigma, L):
    """
    Motor de Diferencias Finitas para la ecuación de Sivashinsky (1993).
    Equivalente al motor espectral (physics_spc.py).
    Utiliza stencils de diferencias finitas centrales para las derivadas espaciales
    en una malla uniforme con condiciones de contorno periódicas.
    """
    N = len(h_field)
    dx = L / N
    
    # Coeficientes según Sivashinsky (1993):
    alpha = 0.5 * beta * (1.0 - Le) - 1.0
    gamma = 4.0
    lambda_param = 0.5 * (1.0 - sigma)

    # 1. Derivadas espaciales de h usando Diferencias Finitas Centrales
    # h_p1: h(i+1), h_m1: h(i-1), etc. (Periodicidad vía np.roll)
    h = h_field
    h_p2 = np.roll(h, -2)
    h_p1 = np.roll(h, -1)
    h_m1 = np.roll(h, 1)
    h_m2 = np.roll(h, 2)

    # h_x -> (h[i+1] - h[i-1]) / (2*dx)
    h_x = (h_p1 - h_m1) / (2.0 * dx)
    
    # h_xx -> (h[i+1] - 2*h[i] + h[i-1]) / (dx**2)
    h_xx = (h_p1 - 2.0 * h + h_m1) / (dx**2)
    
    # h_xxxx -> (h[i+2] - 4*h[i+1] + 6*h[i] - 4*h[i-1] + h[i-2]) / (dx**4)
    h_xxxx = (h_p2 - 4.0 * h_p1 + 6.0 * h - 4.0 * h_m1 + h_m2) / (dx**4)

    # 2. Término de Hilbert I(h)
    # Aunque es un operador no-local, se calcula espectralmente para mantener
    # la consistencia física con physics_spc.py.
    k = 2 * np.pi * np.fft.fftfreq(N, d=dx)
    h_hat = np.fft.fft(h)
    I_h_hat = np.abs(k) * h_hat
    I_h = np.fft.ifft(I_h_hat).real
    
    # 3. Evolución de h (h_t)
    # h_t = -0.5*(h_x^2) - alpha*h_xx - gamma*h_xxxx + lambda*I_h
    # (Siguiendo exactamente la convención euleriana de physics_spc.py)
    h_t = -0.5 * (h_x**2) - alpha * h_xx - gamma * h_xxxx + lambda_param * I_h
    dh_dt = h_t

    # 4. Evolución de u (u_t = (h_t)_x)
    # Calculamos u_t usando diferencias finitas sobre el campo h_t calculado
    h_t_p1 = np.roll(h_t, -1)
    h_t_m1 = np.roll(h_t, 1)
    du_dt = (h_t_p1 - h_t_m1) / (2.0 * dx)

    # 5. Velocidad de la malla (CERO para modo Euleriano)
    v_malla = np.zeros_like(h_field)

    return du_dt, dh_dt, v_malla
