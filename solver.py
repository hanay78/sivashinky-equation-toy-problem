import numpy as np

def calcular_pasos_sts(dt_expl, M, nu):
    """Calcula los sub-pasos tau_j para un ciclo de Super Time Stepping."""
    tau = np.zeros(M)
    for j in range(1, M + 1):
        cos_term = np.cos(np.pi * (2 * j - 1) / (2 * M))
        tau[j - 1] = dt_expl / ((nu - 1) * cos_term + nu + 1)
    return tau
