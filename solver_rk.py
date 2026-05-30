import numpy as np

class InestabilidadError(Exception):
    """Excepción lanzada cuando se detecta inestabilidad numérica."""
    pass

def realizar_paso_rk4(dt, x, u, h_field, m, rho_const, h_smooth_len, beta, Le, sigma, L, t_fisico, T_max, evaluar_sistema_movil, check_estabilidad):
    """Realiza un paso de integración RK4 con control de estabilidad."""
    exito_paso = False

    while not exito_paso:
        try:
            # k1
            k1_u, k1_h, v_m1 = evaluar_sistema_movil(x=x, u=u, h_field=h_field, m=m, rho=rho_const, h_smooth=h_smooth_len, beta=beta, Le=Le, sigma=sigma, L=L)
            k1_x = v_m1

            # k2
            x2 = np.mod(x + 0.5 * dt * k1_x, L)
            u2 = u + 0.5 * dt * k1_u
            h2 = h_field + 0.5 * dt * k1_h
            check_estabilidad(u2, h_smooth_len, dt)
            k2_u, k2_h, v_m2 = evaluar_sistema_movil(x=x2, u=u2, h_field=h2, m=m, rho=rho_const, h_smooth=h_smooth_len, beta=beta, Le=Le, sigma=sigma, L=L)
            k2_x = v_m2

            # k3
            x3 = np.mod(x + 0.5 * dt * k2_x, L)
            u3 = u + 0.5 * dt * k2_u
            h3 = h_field + 0.5 * dt * k2_h
            check_estabilidad(u3, h_smooth_len, dt)
            k3_u, k3_h, v_m3 = evaluar_sistema_movil(x=x3, u=u3, h_field=h3, m=m, rho=rho_const, h_smooth=h_smooth_len, beta=beta, Le=Le, sigma=sigma, L=L)
            k3_x = v_m3

            # k4
            x4 = np.mod(x + dt * k3_x, L)
            u4 = u + dt * k3_u
            h4 = h_field + dt * k3_h
            check_estabilidad(u4, h_smooth_len, dt)
            k4_u, k4_h, v_m4 = evaluar_sistema_movil(x=x4, u=u4, h_field=h4, m=m, rho=rho_const, h_smooth=h_smooth_len, beta=beta, Le=Le, sigma=sigma, L=L)
            k4_x = v_m4

            u_final = u + (dt/6.0) * (k1_u + 2*k2_u + 2*k3_u + k4_u)
            check_estabilidad(u_final, h_smooth_len, dt)

            x = np.mod(x + (dt/6.0) * (k1_x + 2*k2_x + 2*k3_x + k4_x), L)
            u = u_final
            h_field = h_field + (dt/6.0) * (k1_h + 2*k2_h + 2*k3_h + k4_h)
            t_fisico += dt
            exito_paso = True
        except InestabilidadError:
            dt /= 2.0
            if dt < 1e-12:
                print(f"\nInestabilidad crítica en t={t_fisico:.3f}. dt demasiado pequeño. Abortando.")
                t_fisico = T_max
                break

    return x, u, h_field, t_fisico, dt
