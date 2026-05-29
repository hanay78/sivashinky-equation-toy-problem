import numpy as np
import sys
import time

from geometry import ajustar_rho_y_h
from physics2 import evaluar_sistema_movil
from solver import calcular_pasos_sts
from visuals import animar_sivashinsky

class InestabilidadError(Exception):
    """Excepción lanzada cuando se detecta inestabilidad numérica."""
    pass

def simular_ks_sts(metodo='sts'):
    # Parámetros Sivashinsky 1993
    # beta: Zeldovich number, Le: Lewis number, sigma: thermal expansion
    # alpha_eff = 0.5 * beta * (1 - Le) - 1 (inestabilidad termo-difusiva)
    # gamma = 4.0 (estabilización a escalas cortas)
    # lambda = 0.5 * (1 - sigma) (inestabilidad de Darrieus-Landau)
    N, L, beta, Le, sigma, eta, T_max = 256, 100.0, 5.0, 0.3, 0.15, 1.5, 5.5
    M_sts, nu_sts = 4, 0.05
    
    alpha = 0.5 * beta * (1.0 - Le) - 1.0
    gamma = 4.0
    
    x = np.linspace(0, L, N, endpoint=False)
    m = np.ones(N) * (L / N)
    h_smooth_len = np.ones(N) * (eta * L / N)
    
    # Condición inicial: Ruido aleatorio suave en h, y u = h_x compatible
    np.random.seed(42)
    h_field = np.zeros(N)
    u = np.zeros(N)
    for k in range(1, 6): # Suma de 5 modos para un ruido suave
        ak = 0.5 / k
        pk = np.random.rand() * 2 * np.pi
        h_field += ak * np.cos(2 * np.pi * k * x / L + pk)
        u -= ak * (2 * np.pi * k / L) * np.sin(2 * np.pi * k * x / L + pk)
    
    # Inicialización de rho y h (ahora constantes para eficiencia)
    rho_const, h_smooth_len = ajustar_rho_y_h(x, m, h_smooth_len, eta, L)
    
#    def check_estabilidad(u_test):
#        v_max = np.max(np.abs(u_test))
#        if np.isnan(v_max) or v_max > 1e4:
#            raise InestabilidadError()
        
    def check_estabilidad(u_test, h_test, dt_step):
        v_max = np.max(np.abs(u_test))
        h_min = np.min(h_test)        

        dt_new = 0.05 * min(h_min/v_max, h_min**2/(abs(alpha)+1e-2), h_min**4/gamma)
        
        if np.isnan(v_max) or v_max > 1e4 or dt_new < dt_step/2.0:
            raise InestabilidadError()
        
    hist_x, hist_u, hist_h, hist_t = [], [], [], []
    t_fisico, proximo_muestreo, intervalo_muestreo = 0.0, 0.0, 0.05

    print(f"Simulando Sivashinsky 1993 ({metodo.upper()}, N={N}, L={L}, beta={beta}, Le={Le}, sigma={sigma})...", flush=True)
    
    start_time = time.time()
    paso = 0
    while t_fisico < T_max:
        v_max = np.max(np.abs(u)) + 1e-5
        h_min = np.min(h_smooth_len)
        
        if np.isnan(v_max) or v_max > 1e4:
            print(f"\nInestabilidad térmica en t={t_fisico:.3f}. Abortando.", flush=True)
            break
            
        # dt_expl considerando alpha y gamma (aunque alpha sea pequeño o negativo, usamos abs para seguridad)
        dt_expl = 0.05 * min(h_min/v_max, h_min**2/(abs(alpha)+1e-2), h_min**4/gamma)
        
        if metodo == 'sts':
            tau_j = calcular_pasos_sts(dt_expl, M_sts, nu_sts)
            for tau in tau_j:
                # Integración Verlet 2do Orden
                x_mid = np.mod(x + 0.5 * tau * u, L)
                du_dt, dh_dt, _ = evaluar_sistema_movil(x_mid, u, m, rho_const, h_smooth_len, beta, Le, sigma, L)
                
                u_next = u + tau * du_dt
                h_field += tau * dh_dt
                x = np.mod(x_mid + 0.5 * tau * u_next, L)
                u = u_next
                t_fisico += tau
                if t_fisico >= T_max: break
        elif metodo == 'sts4':
            dt = dt_expl
            exito_ciclo = False
            while not exito_ciclo:
                try:
                    # Guardar estado para backtracking
                    x_ini, u_ini, h_ini = x.copy(), u.copy(), h_field.copy()
                    t_ini = t_fisico
                    
                    tau_j = calcular_pasos_sts(dt, M_sts, nu_sts)
                    for tau in tau_j:
                        # RK4 para el sub-paso tau
                        k1_u, k1_h, _ = evaluar_sistema_movil(x, u, m, rho_const, h_smooth_len, beta, Le, sigma, L)
                        k1_x = u
                        
                        x2 = np.mod(x + 0.5 * tau * k1_x, L)
                        u2 = u + 0.5 * tau * k1_u
                        check_estabilidad(u2, h_smooth_len, dt)
                        k2_u, k2_h, _ = evaluar_sistema_movil(x2, u2, m, rho_const, h_smooth_len, beta, Le, sigma, L)
                        k2_x = u2
                        
                        x3 = np.mod(x + 0.5 * tau * k2_x, L)
                        u3 = u + 0.5 * tau * k2_u
                        check_estabilidad(u3, h_smooth_len, dt)
                        k3_u, k3_h, _ = evaluar_sistema_movil(x3, u3, m, rho_const, h_smooth_len, beta, Le, sigma, L)
                        k3_x = u3
                        
                        x4 = np.mod(x + tau * k3_x, L)
                        u4 = u + tau * k3_u
                        check_estabilidad(u4, h_smooth_len, dt)
                        k4_u, k4_h, _ = evaluar_sistema_movil(x4, u4, m, rho_const, h_smooth_len, beta, Le, sigma, L)
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
        elif metodo == 'rk4':
            dt = dt_expl
            exito_paso = False
            while not exito_paso:
                try:
                    # k1
                    k1_u, k1_h, _ = evaluar_sistema_movil(x, u, m, rho_const, h_smooth_len, beta, Le, sigma, L)
                    k1_x = u
                    
                    # k2
                    x2 = np.mod(x + 0.5 * dt * k1_x, L)
                    u2 = u + 0.5 * dt * k1_u
                    check_estabilidad(u2, h_smooth_len, dt)
                    k2_u, k2_h, _ = evaluar_sistema_movil(x2, u2, m, rho_const, h_smooth_len, beta, Le, sigma, L)
                    k2_x = u2
                    
                    # k3
                    x3 = np.mod(x + 0.5 * dt * k2_x, L)
                    u3 = u + 0.5 * dt * k2_u
                    check_estabilidad(u3, h_smooth_len, dt)
                    k3_u, k3_h, _ = evaluar_sistema_movil(x3, u3, m, rho_const, h_smooth_len, beta, Le, sigma, L)
                    k3_x = u3
                    
                    # k4
                    x4 = np.mod(x + dt * k3_x, L)
                    u4 = u + dt * k3_u
                    check_estabilidad(u4, h_smooth_len, dt)
                    k4_u, k4_h, _ = evaluar_sistema_movil(x4, u4, m, rho_const, h_smooth_len, beta, Le, sigma, L)
                    k4_x = u4
                    
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
        
        if t_fisico >= proximo_muestreo:
            hist_x.append(x.copy())
            hist_u.append(u.copy())
            hist_h.append(h_field.copy())
            hist_t.append(t_fisico)
            proximo_muestreo += intervalo_muestreo
            print(f"t: {t_fisico:.3f} | u_max: {v_max:.2f} | h_min: {h_min:.2e}", end="\r", flush=True)
        
        paso += 1
        if paso > 200000: break

    end_time = time.time()
    print(f"\nSimulación finalizada en {end_time - start_time:.2f} segundos.")

    # Llamar a la visualización externa con hist_u
    animar_sivashinsky(hist_x, hist_h, hist_u, hist_t, L, N, t_fisico)

if __name__ == "__main__":
    metodo = 'sts'
    if len(sys.argv) > 1:
        if sys.argv[1].lower() in ['sts', 'rk4', 'sts4']:
            metodo = sys.argv[1].lower()
    
    simular_ks_sts(metodo=metodo)
