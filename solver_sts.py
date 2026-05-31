import numpy as np
from solver_rk import InestabilidadError

def calcular_pasos_sts(dt_expl, M, nu):
    """Calcula los sub-pasos tau_j para un ciclo de Super Time Stepping."""
    tau = np.zeros(M)
    for j in range(1, M + 1):
        cos_term = np.cos(np.pi * (2 * j - 1) / (2 * M))
        tau[j - 1] = dt_expl / ((nu - 1) * cos_term + nu + 1)

    return tau

def realizar_ciclo_sts(dt_expl, M_sts, nu_sts, x, u, h_field, m, rho_const, h_smooth_len, beta, Le, sigma, L, t_fisico, T_max, evaluar_sistema_movil):
    """Realiza un ciclo completo de Super Time Stepping (Euler)."""
    tau_j = calcular_pasos_sts(dt_expl, M_sts, nu_sts)

    for tau in tau_j:
        du_dt, dh_dt, v_malla = evaluar_sistema_movil(x=x, u=u, h_field=h_field, m=m, rho=rho_const, h_smooth=h_smooth_len, beta=beta, Le=Le, sigma=sigma, L=L)

        u_next = u + tau * du_dt
        h_field += tau * dh_dt
        x = np.mod(x + tau * v_malla, L)
        u = u_next
        t_fisico += tau
        if t_fisico >= T_max: break

    return x, u, h_field, t_fisico

