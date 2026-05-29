import numpy as np
import sys
import time

from geometry import ajustar_rho_y_h
from physics2 import evaluar_sistema_movil
import solver_rk
import solver_sts
from visuals import animar_sivashinsky
from initial_condition import establecer_condiciones_iniciales

def simular_ks_sts(metodo='sts2'):

    # beta: Zeldovich number,
    # Le: Lewis number,
    # sigma: thermal expansion
    # alpha_eff = 0.5 * beta * (1 - Le) - 1 (inestabilidad termo-difusiva)
    # gamma = 4.0 (estabilización a escalas cortas)
    # lambda = 0.5 * (1 - sigma) (inestabilidad de Darrieus-Landau)
    
    N = 256
    L = 100.0
    beta = 5.0
    Le = 0.3
    sigma = 0.15
    eta = 1.5
    T_max = 5.5

    # parametros de la ecuacion
    alpha = 0.5 * beta * (1.0 - Le) - 1.0
    gamma = 4.0

    # parametros del sts
    M_sts, nu_sts = 4, 0.05
    
    # parametros sph
    x = np.linspace(0, L, N, endpoint=False)
    m = np.ones(N) * (L / N)
    h_smooth_len = np.ones(N) * (eta * L / N)
    
    # intial conditons
    h_field, u = establecer_condiciones_iniciales(N, L, x)
    
    # Inicialización de rho y h (ahora constantes para eficiencia)
    rho_const, h_smooth_len = ajustar_rho_y_h(x, m, h_smooth_len, eta, L)
    
    def check_estabilidad(u_test, h_test, dt_step):
        v_max = np.max(np.abs(u_test))
        h_min = np.min(h_test)        

        dt_new = 0.05 * min(h_min/v_max, h_min**2/(abs(alpha)+1e-2), h_min**4/gamma)
        
        if np.isnan(v_max) or v_max > 1e4 or dt_new < dt_step/2.0:
            raise solver_rk.InestabilidadError()
        
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
        
        if metodo == 'sts2':
            x, u, h_field, t_fisico = solver_sts.realizar_ciclo_sts2(
                dt_expl=dt_expl, M_sts=M_sts, nu_sts=nu_sts, x=x, u=u, h_field=h_field, 
                m=m, rho_const=rho_const, h_smooth_len=h_smooth_len, 
                beta=beta, Le=Le, sigma=sigma, L=L, 
                t_fisico=t_fisico, T_max=T_max, 
                evaluar_sistema_movil=evaluar_sistema_movil
            )
        elif metodo == 'sts4':
            x, u, h_field, t_fisico, dt_expl = solver_sts.realizar_ciclo_sts4(
                dt=dt_expl, M_sts=M_sts, nu_sts=nu_sts, x=x, u=u, h_field=h_field, 
                m=m, rho_const=rho_const, h_smooth_len=h_smooth_len, 
                beta=beta, Le=Le, sigma=sigma, L=L, 
                t_fisico=t_fisico, T_max=T_max, 
                evaluar_sistema_movil=evaluar_sistema_movil, 
                check_estabilidad=check_estabilidad
            )
        elif metodo == 'rk4':
            x, u, h_field, t_fisico, dt_expl = solver_rk.realizar_paso_rk4(
                dt=dt_expl, x=x, u=u, h_field=h_field, 
                m=m, rho_const=rho_const, h_smooth_len=h_smooth_len, 
                beta=beta, Le=Le, sigma=sigma, L=L, 
                t_fisico=t_fisico, T_max=T_max, 
                evaluar_sistema_movil=evaluar_sistema_movil, 
                check_estabilidad=check_estabilidad
            )
        
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
        if sys.argv[1].lower() in ['sts2', 'rk4', 'sts4']:
            metodo = sys.argv[1].lower()
    
    simular_ks_sts(metodo=metodo)
