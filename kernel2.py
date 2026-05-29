import numpy as np

def kernel_quintico(r, h_input):
    # r puede ser un escalar, vector o matriz de distancias con signo.
    q = np.abs(r) / h_input
    sig_base = 1.0 / (120.0 * h_input)
    h = np.broadcast_to(h_input, q.shape)
    sig = np.broadcast_to(sig_base, q.shape)

    W = np.zeros_like(r)
    dW_dr = np.zeros_like(r)

    m1 = (q < 1)
    if np.any(m1):
        _q = q[m1]
        _sig = sig[m1]
        _h = h[m1]
        W[m1] = _sig * ((3-_q)**5 - 6*(2-_q)**5 + 15*(1-_q)**5)
        dW_dr[m1] = (_sig/_h) * (-5*(3-_q)**4 + 30*(2-_q)**4 - 75*(1-_q)**4)

    m2 = (q >= 1) & (q < 2)
    if np.any(m2):
        _q = q[m2]
        _sig = sig[m2]
        _h = h[m2]
        W[m2] = _sig * ((3-_q)**5 - 6*(2-_q)**5)
        dW_dr[m2] = (_sig/_h) * (-5*(3-_q)**4 + 30*(2-_q)**4)

    m3 = (q >= 2) & (q < 3)
    if np.any(m3):
        _q = q[m3]
        _sig = sig[m3]
        _h = h[m3]
        W[m3] = _sig * ((3-_q)**5)
        dW_dr[m3] = (_sig/_h) * (-5*(3-_q)**4)

    # Para los operadores SPH en 1D necesitamos el gradiente con signo.
    return W, dW_dr * np.sign(r)
