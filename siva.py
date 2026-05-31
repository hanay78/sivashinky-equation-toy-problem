import numpy as np
import sys
import time

from geometry import ajustar_rho_y_h
import solver_rk
import solver_sts
from visuals import animar_sivashinsky
from initial_condition import establecer_condiciones_iniciales

def simular_ks_sts(metodo='sts2', physics='sph', save_video=False):

    if physics == 'sph':
        from physics2 import evaluar_sistema_movil
    elif physics == 'spc':
        from physics_spc import evaluar_sistema_movil        
    elif physics == 'fd':
        from physics_fd import evaluar_sistema_movil
        
    # beta: Zeldovich number,
    # Le: Lewis number,
    # sigma: thermal expansion
    # alpha_eff = 0.5 * beta * (1 - Le) - 1 (inestabilidad termo-difusiva)
    # gamma = 4.0 (estabilización a escalas cortas)
    # lambda = 0.5 * (1 - sigma) (inestabilidad de Darrieus-Landau)
    
#    N = 256
#    N = 512
    N = 1024
    L = 100.0
    beta = 10.0
    Le = 0.3
    sigma = 0.15
    eta = 1.5
#    T_max = 25.
    T_max = 100.
    CFL = 0.5
    
    # parametros de la ecuacion
    alpha = 0.5 * beta * (1.0 - Le) - 1.0
    gamma = 4.0

    # parametros del sts
    M_sts, nu_sts = 10, 0.005
    
    # parametros sph
    x = np.linspace(0, L, N, endpoint=False)
    m = np.ones(N) * (L / N)
    h_smooth_len = np.ones(N) * (eta * L / N)
    
    # intial conditons
    h_field, u = establecer_condiciones_iniciales(N, L, x)
    
    # Inicialización de rho y h (ahora constantes para eficiencia)
    rho_const, h_smooth_len = ajustar_rho_y_h(x, m, h_smooth_len, eta, L)

    # Escala de resolución para el paso de tiempo
    ds_min = (L/N) if (physics == 'fd' or physics == 'spc')  else np.min(h_smooth_len)

    def calcular_dt_expl(v_max):
        dt_adv = ds_min / v_max
        dt_diff = ds_min**2 / (abs(alpha) + 1e-2)
        dt_fourth =  ds_min**4 / (abs(gamma)+1e-2)
        
        if physics == 'fd':
            dt_diff *= 0.5
            dt_fourth *= 0.125
            
        elif physics == 'spc':
            fac=1.0/np.pi

            dt_adv *= fac
            dt_diff *= fac**2
            dt_fourth *= fac**4

        elif physics == 'sph':
             dt_adv *= 0.25
             dt_diff *= 0.125
             dt_fourth *= 0.0125
              
        return CFL * min(dt_adv, dt_diff, dt_fourth)

    def check_estabilidad(u_test, h_test, dt_step):
        v_max = np.max(np.abs(u_test))
        dt_new = calcular_dt_expl(v_max)

        if np.isnan(v_max) or v_max > 1e4 or dt_new < dt_step/2.0:
            raise solver_rk.InestabilidadError()

    hist_x, hist_u, hist_h, hist_t = [], [], [], []
    t_fisico, proximo_muestreo, intervalo_muestreo = 0.0, 0.0, 0.05

    print(f"Simulando Sivashinsky 1993 ({metodo.upper()}, Physics={physics.upper()}, N={N}, L={L}, beta={beta}, Le={Le}, sigma={sigma})...", flush=True)

    start_time = time.time()
    paso = 0
    while t_fisico < T_max:
        v_max = np.max(np.abs(u)) + 1e-5
        h_min = np.min(np.abs(h_smooth_len))
        
        if np.isnan(v_max) or v_max > 1e4:
            print(f"\nInestabilidad térmica en t={t_fisico:.3f}. Abortando.", flush=True)
            break

        # dt_expl considerando alpha y gamma
        dt_expl = calcular_dt_expl(v_max)
        
        if metodo == 'sts':
            x, u, h_field, t_fisico = solver_sts.realizar_ciclo_sts(
                dt_expl=dt_expl, M_sts=M_sts, nu_sts=nu_sts, x=x, u=u, h_field=h_field, 
                m=m, rho_const=rho_const, h_smooth_len=h_smooth_len, 
                beta=beta, Le=Le, sigma=sigma, L=L, 
                t_fisico=t_fisico, T_max=T_max, 
                evaluar_sistema_movil=evaluar_sistema_movil
            )
        elif metodo == 'sts2':
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
       
    end_time = time.time()
    print(f"\nSimulación finalizada en {end_time - start_time:.2f} segundos.")

    # Llamar a la visualización externa con hist_u
    animar_sivashinsky(hist_x, hist_h, hist_u, hist_t, L, N, t_fisico, save_video=save_video)

if __name__ == "__main__":
    metodo = 'sts2'
    physics = 'sph'
    save_video = False
    
    if len(sys.argv) > 1:
        for arg in sys.argv[1:]:
            arg_low = arg.lower()
            if arg_low in ['sts', 'sts2', 'rk4', 'sts4']:
                metodo = arg_low
            elif arg_low in ['sph', 'fd', 'spc']:
                physics = arg_low
            elif arg_low in ['video',]:
                save_video = True
    
    simular_ks_sts(metodo=metodo, physics=physics, save_video=save_video)
