import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def animar_sivashinsky(hist_x, hist_h, hist_u, hist_t, L, N, t_final):
    """Genera una visualización de tres paneles en horizontal (h, u, trayectorias)."""
    print("\nVisualizando dinámica completa (h, u, trayectorias)...", flush=True)

    fig, (ax_h, ax_u, ax_st) = plt.subplots(1, 3, figsize=(18, 6))
    
    # 1. Panel: Interface h(x)
    line_h, = ax_h.plot([], [], 'r-', lw=2, label='Altura h')
    dots_h = ax_h.scatter([], [], c='black', s=5, alpha=0.5)
    ax_h.set_xlim(0, L)
    ax_h.set_ylim(-5, 5)
    ax_h.set_title('Interface h(x)')
    ax_h.set_xlabel('x')
    ax_h.set_ylabel('h')
    time_text = ax_h.text(0.05, 0.9, '', transform=ax_h.transAxes)

    # 2. Panel: Velocidad u(x)
    line_u, = ax_u.plot([], [], 'b-', lw=1.5, alpha=0.6, label='Velocidad u')
    dots_u = ax_u.scatter([], [], c='black', s=5, alpha=0.5)
    ax_u.set_xlim(0, L)
    ax_u.set_ylim(-5, 5)
    ax_u.set_title('Velocidad u(x)')
    ax_u.set_xlabel('x')
    ax_u.set_ylabel('u')

    # 3. Panel: Trayectorias (Space-Time)
    px = np.array(hist_x).flatten()
    pt = np.array([[t]*N for t in hist_t]).flatten()
    ax_st.scatter(px, pt, color='black', s=5., edgecolors='none', alpha=0.2)
    ax_st.set_xlim(0, L)
    ax_st.set_ylim(0, t_final)
    ax_st.set_title('Trayectorias (Space-Time)')
    ax_st.set_xlabel('x')
    ax_st.set_ylabel('t')

    # Precalculo de límites para optimizar la animación
    h_mins = np.min(hist_h, axis=1)
    h_maxs = np.max(hist_h, axis=1)
    u_mins = np.min(hist_u, axis=1)
    u_maxs = np.max(hist_u, axis=1)
    
    # Calcular márgenes (buffer) de forma vectorizada
    dh = np.maximum(0.5, 0.1 * (h_maxs - h_mins))
    du = np.maximum(0.5, 0.1 * (u_maxs - u_mins))

    # Variables para control de actualización de ejes
    current_h_lims = [None, None]
    current_u_lims = [None, None]

    def init():
        line_h.set_data([], [])
        dots_h.set_offsets(np.empty((0, 2)))
        line_u.set_data([], [])
        dots_u.set_offsets(np.empty((0, 2)))
        time_text.set_text('')
        return line_h, dots_h, line_u, dots_u, time_text

    def animate(i):
        nonlocal current_h_lims, current_u_lims
        xi = hist_x[i]
        hi = hist_h[i]
        ui = hist_u[i]
        idx = np.argsort(xi)
        
        # Actualizar datos de líneas y puntos
        line_h.set_data(xi[idx], hi[idx])
        dots_h.set_offsets(np.column_stack((xi, hi)))
        line_u.set_data(xi[idx], ui[idx])
        dots_u.set_offsets(np.column_stack((xi, ui)))

        # Solo actualizar límites si el cambio es significativo (>5%) para evitar redraws constantes
        new_h_min, new_h_max = h_mins[i] - dh[i], h_maxs[i] + dh[i]
        new_u_min, new_u_max = u_mins[i] - du[i], u_maxs[i] + du[i]

        if current_h_lims[0] is None or \
           abs(new_h_min - current_h_lims[0]) > 0.05 * abs(current_h_lims[0] or 1) or \
           abs(new_h_max - current_h_lims[1]) > 0.05 * abs(current_h_lims[1] or 1):
            ax_h.set_ylim(new_h_min, new_h_max)
            current_h_lims = [new_h_min, new_h_max]

        if current_u_lims[0] is None or \
           abs(new_u_min - current_u_lims[0]) > 0.05 * abs(current_u_lims[0] or 1) or \
           abs(new_u_max - current_u_lims[1]) > 0.05 * abs(current_u_lims[1] or 1):
            ax_u.set_ylim(new_u_min, new_u_max)
            current_u_lims = [new_u_min, new_u_max]
        
        time_text.set_text(f't = {hist_t[i]:.2f}')
        return line_h, dots_h, line_u, dots_u, time_text

    # Se desactiva blit=True para permitir la actualización de los ejes (ticks y límites)
    ani = FuncAnimation(fig, animate, frames=len(hist_t), init_func=init, blit=False, interval=40)

    plt.tight_layout()

    # 1. Guardar imagen estática del resultado
    print("Guardando imagen estática en 'siva_result.png'...", flush=True)
    plt.savefig('siva_result.png', dpi=150)

    # 2. Guardar la animación como video
    # try:
    #     print("Guardando animación en 'siva_animation.mp4' (esto puede tardar)...", flush=True)
    #     ani.save('siva_animation.mp4', writer='ffmpeg', fps=20)
    #     print("Animación guardada correctamente.", flush=True)
    # except Exception as e:
    #     print(f"Nota: No se pudo guardar el MP4 (¿Falta ffmpeg?). Error: {e}", flush=True)

    plt.show()
    return ani
