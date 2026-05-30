import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def animar_sivashinsky(hist_x, hist_h, hist_u, hist_t, L, N, t_final):
    """
    Versión de alto rendimiento para la visualización de la ecuación de Sivashinsky.
    Utiliza blitting y escalado global para máxima fluidez.
    """
    print("\nVisualizando dinámica completa (h, u, trayectorias)...", flush=True)

    fig, (ax_h, ax_u, ax_st) = plt.subplots(1, 3, figsize=(18, 6))
    
    # 0. Precálculo de límites globales para evitar redibujado de ejes (necesario para blit=True)
    h_min_g, h_max_g = np.min(hist_h), np.max(hist_h)
    u_min_g, u_max_g = np.min(hist_u), np.max(hist_u)
    
    # Margen del 10%
    dh = max(0.5, 0.1 * (h_max_g - h_min_g))
    du = max(0.5, 0.1 * (u_max_g - u_min_g))

    # 1. Panel: Interface h(x)
    line_h, = ax_h.plot([], [], 'r-', lw=2, label='Altura h')
    # Usar plot con markers es mucho más rápido que scatter para animaciones
    dots_h, = ax_h.plot([], [], 'ko', ms=2, alpha=0.3)
    ax_h.set_xlim(0, L)
    ax_h.set_ylim(h_min_g - dh, h_max_g + dh)
    ax_h.set_title('Interface h(x)')
    ax_h.set_xlabel('x')
    ax_h.set_ylabel('h')
    time_text = ax_h.text(0.05, 0.9, '', transform=ax_h.transAxes)

    # 2. Panel: Velocidad u(x)
    line_u, = ax_u.plot([], [], 'b-', lw=1.5, alpha=0.6, label='Velocidad u')
    dots_u, = ax_u.plot([], [], 'ko', ms=2, alpha=0.3)
    ax_u.set_xlim(0, L)
    ax_u.set_ylim(u_min_g - du, u_max_g + du)
    ax_u.set_title('Velocidad u(x)')
    ax_u.set_xlabel('x')
    ax_u.set_ylabel('u')

    # 3. Panel: Trayectorias (Space-Time) - Optimizado con muestreo
    step = max(1, len(hist_t) // 1000) # Limitar a ~1000 instantes para no saturar
    px = np.array(hist_x[::step]).flatten()
    pt = np.array([[t]*N for t in hist_t[::step]]).flatten()
    ax_st.scatter(px, pt, color='black', s=1.0, edgecolors='none', alpha=0.1)
    ax_st.set_xlim(0, L)
    ax_st.set_ylim(0, t_final)
    ax_st.set_title('Trayectorias (Space-Time)')
    ax_st.set_xlabel('x')
    ax_st.set_ylabel('t')

    def init():
        line_h.set_data([], [])
        dots_h.set_data([], [])
        line_u.set_data([], [])
        dots_u.set_data([], [])
        time_text.set_text('')
        return line_h, dots_h, line_u, dots_u, time_text

    def animate(i):
        xi = hist_x[i]
        hi = hist_h[i]
        ui = hist_u[i]
        
        # En FD/SPC los puntos están ordenados, pero en SPH no.
        # Ordenamos solo para la línea continua.
        idx = np.argsort(xi)
        
        # Actualizar h
        line_h.set_data(xi[idx], hi[idx])
        dots_h.set_data(xi, hi)
        
        # Actualizar u
        line_u.set_data(xi[idx], ui[idx])
        dots_u.set_data(xi, ui)
        
        time_text.set_text(f't = {hist_t[i]:.2f}')
        return line_h, dots_h, line_u, dots_u, time_text

    # blit=True es la clave para la velocidad
    ani = FuncAnimation(fig, animate, frames=len(hist_t), init_func=init, blit=True, interval=15)

    plt.tight_layout()

    # 1. Guardar imagen estática del resultado
    print("Guardando imagen estática en 'siva_result.png'...", flush=True)
    plt.savefig('siva_result.png', dpi=150)

    plt.show()
    return ani
