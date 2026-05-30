import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def animar_sivashinsky(hist_x, hist_h, hist_u, hist_t, L, N, t_final):
    """
    Visualización en 2x2:
    - Superior Izquierda: h(x) con escala adaptativa.
    - Superior Derecha: u(x) con escala adaptativa.
    - Inferior Izquierda: h(x) con escala FIJA (Referencia).
    - Inferior Derecha: Trayectorias Space-Time (Estático).
    """
    print("\nIniciando visualización 2x2 de alto rendimiento...", flush=True)

    plt.close('all')
    fig, ((ax_h, ax_u), (ax_fix, ax_st)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. Precálculo de límites por frame para las escalas adaptativas
    h_mins = np.min(hist_h, axis=1)
    h_maxs = np.max(hist_h, axis=1)
    u_mins = np.min(hist_u, axis=1)
    u_maxs = np.max(hist_u, axis=1)
    
    # Límites globales para el panel fijo
    h_min_g, h_max_g = np.min(hist_h), np.max(hist_h)
    dh_g = max(0.5, 0.1 * (h_max_g - h_min_g))

    # 2. Configuración de Paneles (Objetos animados)
    
    # Panel h adaptativo
    line_h, = ax_h.plot([], [], 'r-', lw=2, animated=True)
    dots_h, = ax_h.plot([], [], 'ko', ms=1.5, alpha=0.2, animated=True)
    ax_h.set_xlim(0, L)
    ax_h.set_ylim(h_mins[0] - 1.0, h_maxs[0] + 1.0)
    ax_h.set_title('Interface h(x) [Adaptativa]')
    time_text = ax_h.text(0.05, 0.9, '', transform=ax_h.transAxes, fontweight='bold', animated=True)

    # Panel u adaptativo
    line_u, = ax_u.plot([], [], 'b-', lw=1.5, alpha=0.6, animated=True)
    dots_u, = ax_u.plot([], [], 'ko', ms=1.5, alpha=0.2, animated=True)
    ax_u.set_xlim(0, L)
    ax_u.set_ylim(u_mins[0] - 1.0, u_maxs[0] + 1.0)
    ax_u.set_title('Velocidad u(x) [Adaptativa]')

    # Panel h FIJO (Referencia)
    line_fix, = ax_fix.plot([], [], 'g-', lw=2, animated=True)
    dots_fix, = ax_fix.plot([], [], 'ko', ms=1.5, alpha=0.2, animated=True)
    ax_fix.set_xlim(0, L)
    ax_fix.set_ylim(h_min_g - dh_g, h_max_g + dh_g) # Escala fija desde el inicio
    ax_fix.set_title(f'Interface h(x) [Fija: {h_min_g:.1f}, {h_max_g:.1f}]')
    ax_fix.set_xlabel('x')
    ax_fix.set_ylabel('h')

    # Panel Space-Time (Estático)
    step_st = max(1, len(hist_t) // 1000)
    px = np.array(hist_x[::step_st]).flatten()
    pt = np.array([[t]*N for t in hist_t[::step_st]]).flatten()
    ax_st.scatter(px, pt, color='black', s=0.5, edgecolors='none', alpha=0.1)
    ax_st.set_xlim(0, L)
    ax_st.set_ylim(0, t_final)
    ax_st.set_title('Space-Time')
    ax_st.set_xlabel('x')
    ax_st.set_ylabel('t')

    def init():
        line_h.set_data([], [])
        dots_h.set_data([], [])
        line_u.set_data([], [])
        dots_u.set_data([], [])
        line_fix.set_data([], [])
        dots_fix.set_data([], [])
        time_text.set_text('')
        return line_h, dots_h, line_u, dots_u, line_fix, dots_fix, time_text

    def animate(i):
        xi = hist_x[i]
        hi = hist_h[i]
        ui = hist_u[i]
        idx = np.argsort(xi)
        
        # Lógica de expansión para los paneles adaptativos
        h_lim = ax_h.get_ylim()
        u_lim = ax_u.get_ylim()
        needs_redraw = False
        
        if hi.min() < h_lim[0] or hi.max() > h_lim[1]:
            margin = 0.3 * (hi.max() - hi.min() + 1)
            ax_h.set_ylim(hi.min() - margin, hi.max() + margin)
            needs_redraw = True
            
        if ui.min() < u_lim[0] or ui.max() > u_lim[1]:
            margin = 0.3 * (ui.max() - ui.min() + 1)
            ax_u.set_ylim(ui.min() - margin, ui.max() + margin)
            needs_redraw = True

        if needs_redraw:
            fig.canvas.draw()

        # Actualizar todos los datos
        line_h.set_data(xi[idx], hi[idx])
        dots_h.set_data(xi, hi)
        
        line_u.set_data(xi[idx], ui[idx])
        dots_u.set_data(xi, ui)
        
        line_fix.set_data(xi[idx], hi[idx])
        dots_fix.set_data(xi, hi)
        
        time_text.set_text(f't = {hist_t[i]:.2f}')

        return line_h, dots_h, line_u, dots_u, line_fix, dots_fix, time_text

    ani = FuncAnimation(fig, animate, frames=len(hist_t), init_func=init, blit=True, interval=10)

    plt.tight_layout()
    print("Guardando imagen 'siva_result.png'...", flush=True)
    plt.savefig('siva_result.png', dpi=120)

    # 2. Guardar la animación como video (Alta Eficiencia y Resolución)
    # try:
    #     # Calculamos un salto de frames para optimizar el tiempo de renderizado
    #     # Apuntamos a un máximo de ~600 frames para un vídeo de ~20s a 30fps
    #     total_steps = len(hist_t)
    #     n_frames_objetivo = 600
    #     skip = max(1, total_steps // n_frames_objetivo)
        
    #     print(f"Exportando {total_steps//skip} frames a 'siva_animation.mp4' (Alta Calidad)...", flush=True)
        
    #     # Creamos una versión filtrada de la animación solo para el guardado
    #     # Esto evita renderizar frames redundantes que el ojo no apreciaría a 30fps
    #     ani.save('siva_animation.mp4', 
    #              writer='ffmpeg', 
    #              fps=30, 
    #              dpi=120, # Mantenemos alta resolución
    #              extra_args=[
    #                  '-vcodec', 'libx264', 
    #                  '-preset', 'ultrafast', # Prioriza velocidad de encoding
    #                  '-threads', '0',        # Usa todos los núcleos disponibles
    #                  '-crf', '18',           # Calidad visual casi idéntica a la original
    #                  '-pix_fmt', 'yuv420p'
    #              ])
    #     print("Vídeo guardado correctamente.", flush=True)
    # except Exception as e:
    #     print(f"Nota: No se pudo guardar el MP4. Error: {e}", flush=True)

    plt.show()
    return ani
