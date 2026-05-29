import numpy as np
from solver_rk import InestabilidadError

def calcular_pasos_sts(dt_expl, M, nu):
    """Calcula los sub-pasos tau_j para un ciclo de Super Time Stepping."""
    tau = np.zeros(M)
    for j in range(1, M + 1):
        cos_term = np.cos(np.pi * (2 * j - 1) / (2 * M))
        tau[j - 1] = dt_expl / ((nu - 1) * cos_term + nu + 1)

    return tau

def realizar_ciclo_sts2(dt_expl, M_sts, nu_sts, x, u, h_field, m, rho_const, h_smooth_len, beta, Le, sigma, L, t_fisico, T_max, evaluar_sistema_movil):
    """Realiza un ciclo completo de Super Time Stepping."""
    tau_j = calcular_pasos_sts(dt_expl, M_sts, nu_sts)

    for tau in tau_j:
        # Integración Verlet 2do Orden
        x_mid = np.mod(x + 0.5 * tau * u, L)
        du_dt, dh_dt, _ = evaluar_sistema_movil(x=x_mid, u=u, m=m, rho=rho_const, h=h_smooth_len, beta=beta, Le=Le, sigma=sigma, L=L)
        
        u_next = u + tau * du_dt
        h_field += tau * dh_dt
        x = np.mod(x_mid + 0.5 * tau * u_next, L)
        u = u_next
        t_fisico += tau
        if t_fisico >= T_max: break
        
    return x, u, h_field, t_fisico

def realizar_ciclo_sts4(dt, M_sts, nu_sts, x, u, h_field, m, rho_const, h_smooth_len, beta, Le, sigma, L, t_fisico, T_max, evaluar_sistema_movil, check_estabilidad):
    """Realiza un ciclo de Super Time Stepping de 4to orden (STS4)."""
    exito_ciclo = False
    while not exito_ciclo:
        try:
            # Guardar estado para backtracking
            x_ini, u_ini, h_ini = x.copy(), u.copy(), h_field.copy()
            t_ini = t_fisico
            
            tau_j = calcular_pasos_sts(dt, M_sts, nu_sts)
            for tau in tau_j:
                # RK4 para el sub-paso tau
                k1_u, k1_h, _ = evaluar_sistema_movil(x=x, u=u, m=m, rho=rho_const, h=h_smooth_len, beta=beta, Le=Le, sigma=sigma, L=L)
                k1_x = u
                
                x2 = np.mod(x + 0.5 * tau * k1_x, L)
                u2 = u + 0.5 * tau * k1_u
                check_estabilidad(u2, h_smooth_len, dt)
                k2_u, k2_h, _ = evaluar_sistema_movil(x=x2, u=u2, m=m, rho=rho_const, h=h_smooth_len, beta=beta, Le=Le, sigma=sigma, L=L)
                k2_x = u2
                
                x3 = np.mod(x + 0.5 * tau * k2_x, L)
                u3 = u + 0.5 * tau * k2_u
                check_estabilidad(u3, h_smooth_len, dt)
                k3_u, k3_h, _ = evaluar_sistema_movil(x=x3, u=u3, m=m, rho=rho_const, h=h_smooth_len, beta=beta, Le=Le, sigma=sigma, L=L)
                k3_x = u3
                
                x4 = np.mod(x + tau * k3_x, L)
                u4 = u + tau * k3_u
                check_estabilidad(u4, h_smooth_len, dt)
                k4_u, k4_h, _ = evaluar_sistema_movil(x=x4, u=u4, m=m, rho=rho_const, h=h_smooth_len, beta=beta, Le=Le, sigma=sigma, L=L)
                k4_x = u4
                
                u_next = u + (tau/6.0) * (k1_u + 2*k2_u + 2*k3_u + k4_u)
                check_estabilidad(u_next, h_smooth_len, dt)
                
                x = np.mod(x + (tau/6.0) * (k1_x + 2*k2_x + 2*k3_x + k4_x), L)
                u = u_next
                h_field += (tau/6.0) * (k1_h + 2*k2_h + 2*k3_h + k4_h)
                t_fisico += tau
                if t_fisico >= T_max: break
            exito_ciclo = True

        except InestabilidadError:
            x, u, h_field = x_ini, u_ini, h_ini
            t_fisico = t_ini
            dt /= 2.0
            if dt < 1e-12:
                print(f"\nInestabilidad crítica en sts4 en t={t_fisico:.3f}. Abortando.")
                t_fisico = T_max
                break
            
    return x, u, h_field, t_fisico, dt
