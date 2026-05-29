import numpy as np

def establecer_condiciones_iniciales(N, L, x):
    """
    Establece las condiciones iniciales para la simulación de Kuramoto-Sivashinsky.
    
    Args:
        N (int): Número de puntos de la malla.
        L (float): Longitud del dominio.
        x (ndarray): Coordenadas de la malla.
        
    Returns:
        tuple: (h_field, u) donde h_field es el campo de altura y u es el campo de velocidad.
    """
    # Condición inicial: Ruido aleatorio suave en h, y u = h_x compatible
    np.random.seed(42)
    h_field = np.zeros(N)
    u = np.zeros(N)
    for k in range(1, 6): # Suma de 5 modos para un ruido suave
        ak = 0.5 / k
        pk = np.random.rand() * 2 * np.pi
        h_field += ak * np.cos(2 * np.pi * k * x / L + pk)
        u -= ak * (2 * np.pi * k / L) * np.sin(2 * np.pi * k * x / L + pk)
    return h_field, u
