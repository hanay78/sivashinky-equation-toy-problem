import numpy as np

def evaluar_sistema_movil(x, u, h_field, m, rho, h_smooth, beta, Le, sigma, L):
    """
    Motor Espectral Puro para la ecuación de Sivashinsky (1993).
    Utiliza FFT para todas las derivadas espaciales y el término de Hilbert.
    Esto es mucho más estable y preciso que Diferencias Finitas en malla uniforme.
    """
    N = len(h_field)
    dx = L / N
    
    # Coeficientes según Sivashinsky (1993):
    alpha = 0.5 * beta * (1.0 - Le) - 1.0
    gamma = 4.0
    lambda_param = 0.5 * (1.0 - sigma)

    # 1. Espacio de Fourier
    # k = [0, 1, ..., N/2, -N/2, ..., -1] * 2*pi / L
    k = 2 * np.pi * np.fft.fftfreq(N, d=dx)
    h_hat = np.fft.fft(h_field)
    
    # 2. Derivadas en espacio de Fourier
    # h_x -> i*k * h_hat
    h_x_hat = 1j * k * h_hat
    # h_xx -> -k^2 * h_hat
    h_xx_hat = -(k**2) * h_hat
    # h_xxxx -> k^4 * h_hat
    h_xxxx_hat = (k**4) * h_hat
    # I(h) -> |k| * h_hat
    I_h_hat = np.abs(k) * h_hat
    
    # 3. Regreso al espacio real
    h_x = np.fft.ifft(h_x_hat).real
    h_xx = np.fft.ifft(h_xx_hat).real
    h_xxxx = np.fft.ifft(h_xxxx_hat).real
    I_h = np.fft.ifft(I_h_hat).real
    
    # 4. Evolución de h (h_t)
    # h_t = -0.5*(h_x^2) - alpha*h_xx - gamma*h_xxxx + lambda*I_h
    # Nota: No incluimos el 1/2pi extra aquí, la FFT ya es autoconsistente.
    # Si la física requiere un factor de escala para lambda, se ajusta externamente.
    h_t = -0.5 * (h_x**2) - alpha * h_xx - gamma * h_xxxx + lambda_param * I_h
    dh_dt = h_t

    # 5. Evolución de u (u_t = (h_t)_x)
    # Calculamos u_t espectralmente desde h_t para máxima consistencia
    h_t_hat = np.fft.fft(h_t)
    u_t_hat = 1j * k * h_t_hat
    du_dt = np.fft.ifft(u_t_hat).real

    # 6. Velocidad de la malla (CERO para modo Euleriano/Espectral)
    v_malla = np.zeros_like(h_field)

    return du_dt, dh_dt, v_malla
