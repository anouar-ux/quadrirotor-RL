"""
Animation (GIF) de la poursuite de la voiture par le drone et de l'atterrissage, à partir d'un vol ENREGISTRÉ.

    python python/evaluation/animation_mission.py                      # vol Simulink, agents, g_noise = 0,1
    python python/evaluation/animation_mission.py --vol rl_g000        # vol Simulink sans bruit
    python python/evaluation/animation_mission.py --source python      # vol du simulateur Python (sim_mission_vols_cap.npz)

Sortie : resultats/mission/animation_mission.gif (temps réel, 10 images/s ; de la fin du scan au posé + 2 s).
Trois vues :
  - à gauche, vue de dessus centrée entre le drone et la voiture, zoom qui suit leur distance : voiture orientée selon son cap, drone (4 rotors, flèche
    = son cap psi), traces ; la couleur du drone indique la phase ;
  - en haut à droite, vue d'ensemble de la trajectoire ;
  - en bas à droite, altitude z(t) et écart horizontal à la voiture, avec le curseur du temps.
"""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import animation, patches, transforms  # noqa: E402
from scipy.io import loadmat  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # dossier python/ (chemins.py)
from chemins import EI, RESULTATS, SIMULINK_RL  # noqa: E402

TS = 0.05
PHASES = ("stabilisation", "montée", "scan", "poursuite", "descente", "posé")
COUL = {2: "tab:purple", 3: "tab:blue", 4: "tab:orange", 5: "tab:green"}
L_VOIT, W_VOIT = 0.45, 0.22        # gabarit de la voiture dessinée [m]
R_DRONE = 0.12                     # demi-envergure du drone dessiné [m]


def charger(source, nom_vol):
    """t, x, y, z, psi, phase du drone et table de la voiture."""
    voit = loadmat(SIMULINK_RL / "mission" / "voiture.mat", squeeze_me=True)
    if source == "simulink":
        S = loadmat(SIMULINK_RL / "mission" / "resultats_mission_cap.mat", squeeze_me=True, struct_as_record=False)
        L = getattr(S["R"], nom_vol)
        d = dict(t=L[:, 0], x=L[:, 1], y=L[:, 2], z=L[:, 3], psi=L[:, 9], phase=L[:, 23])
        titre = f"Simulink (UAV_model), {nom_vol.replace('rl_', 'agents, ').replace('g000', 'sans bruit').replace('g100', 'g_noise = 0,1')}"
    else:
        Z = np.load(RESULTATS / "mission" / "sim_mission_vols_cap.npz")
        v, y = Z["agents, g=0,1_vrai"], Z["agents, g=0,1_y"]
        d = dict(t=Z["agents, g=0,1_v"][:, 0], x=v[:, 0], y=v[:, 1], z=v[:, 2], psi=v[:, 8], phase=y[:, 4])
        titre = "simulateur Python assemblé, agents, g_noise = 0,1"
    return d, voit, titre


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", choices=("simulink", "python"), default="simulink")
    p.add_argument("--vol", default="rl_g100")
    p.add_argument("--pas", type=int, default=2, help="un pas de simulation sur N par image (2 : 10 images/s)")
    p.add_argument("--sortie", default=str(RESULTATS / "mission" / "animation_mission.gif"))
    a = p.parse_args()
    d, V, titre = charger(a.source, a.vol)
    t, ph = d["t"], d["phase"]
    i0 = int(np.argmax(ph >= 3)) - int(1.0 / TS)                     # 1 s avant la fin du scan
    i1 = min(int(np.argmax(ph >= 5)) + int(2.0 / TS), len(t) - 1)    # 2 s après le posé
    idx = np.arange(i0, i1 + 1, a.pas)
    kv = np.minimum(np.round(t / TS).astype(int), len(V["t"]) - 1)
    xv, yv, cap = V["x"][kv], V["y"][kv], V["cap"][kv]
    eh = np.hypot(d["x"] - xv, d["y"] - yv)

    fig = plt.figure(figsize=(11, 6.2))
    gs = fig.add_gridspec(2, 2, width_ratios=(1.25, 1), height_ratios=(1, 1))
    axz = fig.add_subplot(gs[:, 0]); axg = fig.add_subplot(gs[0, 1]); axh = fig.add_subplot(gs[1, 1])
    fig.suptitle(f"Mission finale : poursuite de la voiture et atterrissage ({titre})", fontsize=11)
    # vue d'ensemble
    axg.plot(V["x"][kv[i0]:kv[i1]], V["y"][kv[i0]:kv[i1]], color="0.6", lw=4, alpha=0.5, label="voiture")
    axg.plot(d["x"][:i1], d["y"][:i1], color="tab:blue", lw=1, label="drone")
    pg_v, = axg.plot([], [], "s", color="0.3", ms=6); pg_d, = axg.plot([], [], "o", color="tab:blue", ms=5)
    axg.set_aspect("equal"); axg.grid(True); axg.legend(fontsize=8, loc="upper left"); axg.set_title("vue d'ensemble", fontsize=9)
    axg.set_xlabel("x [m]", fontsize=8); axg.set_ylabel("y [m]", fontsize=8); axg.tick_params(labelsize=8)
    # altitude et écart
    axh.plot(t[i0:i1], d["z"][i0:i1], color="tab:blue", label="altitude z [m]")
    axh.plot(t[i0:i1], np.minimum(eh[i0:i1], 1.0), color="tab:red", lw=1, label="écart horizontal [m] (borné à 1)")
    axh.axhline(0.05, color="tab:red", ls=":", lw=0.8)
    cur = axh.axvline(t[i0], color="k", lw=1)
    axh.set_xlabel("t [s]", fontsize=8); axh.grid(True); axh.legend(fontsize=8, loc="upper right"); axh.tick_params(labelsize=8)
    axh.set_ylim(-0.05, 1.05)
    # vue qui suit la voiture
    axz.set_aspect("equal"); axz.grid(True); axz.set_xlabel("x [m]"); axz.set_ylabel("y [m]")
    trace_v, = axz.plot([], [], color="0.6", lw=3, alpha=0.5)
    trace_d, = axz.plot([], [], color="tab:blue", lw=1.2)
    voit = patches.Rectangle((-L_VOIT / 2, -W_VOIT / 2), L_VOIT, W_VOIT, fc="0.35", ec="k", zorder=3)
    plateforme = patches.Rectangle((-0.1, -0.08), 0.2, 0.16, fc="w", ec="k", lw=0.8, zorder=4)
    axz.add_patch(voit); axz.add_patch(plateforme)
    bras = [axz.plot([], [], color="k", lw=2, zorder=6)[0] for _ in range(2)]
    rotors = [patches.Circle((0, 0), 0.045, fc="tab:blue", ec="k", zorder=7) for _ in range(4)]
    for r in rotors:
        axz.add_patch(r)
    fleche = axz.annotate("", xy=(0, 0), xytext=(0, 0), arrowprops=dict(arrowstyle="->", lw=2, color="tab:red"), zorder=8)
    ombre = patches.Circle((0, 0), R_DRONE, fc="k", alpha=0.15, zorder=2)
    axz.add_patch(ombre)
    texte = axz.text(0.02, 0.98, "", transform=axz.transAxes, va="top", fontsize=10,
                     bbox=dict(fc="w", ec="0.7", alpha=0.9), zorder=9)

    def image(n):
        i = idx[n]
        cx, cy = xv[i], yv[i]
        mx, my = (cx + d["x"][i]) / 2, (cy + d["y"][i]) / 2           # vue centrée entre drone et voiture,
        demi = max(1.2, 0.55 * eh[i] + 0.8)                                # zoom qui suit leur distance
        axz.set_xlim(mx - demi, mx + demi); axz.set_ylim(my - demi, my + demi)
        trace_v.set_data(xv[i0:i + 1], yv[i0:i + 1]); trace_d.set_data(d["x"][i0:i + 1], d["y"][i0:i + 1])
        tr = transforms.Affine2D().rotate(cap[i]).translate(cx, cy) + axz.transData
        voit.set_transform(tr); plateforme.set_transform(tr)
        x, y, s = d["x"][i], d["y"][i], 1.0 + 1.2 * max(d["z"][i], 0.0)       # le drone « grossit » avec l'altitude
        c, si = np.cos(d["psi"][i]), np.sin(d["psi"][i])
        coul = COUL.get(int(ph[i]), "tab:blue")
        for j, ang in enumerate((np.pi / 4, -np.pi / 4)):
            ca, sa = np.cos(d["psi"][i] + ang), np.sin(d["psi"][i] + ang)
            bras[j].set_data([x - s * R_DRONE * ca, x + s * R_DRONE * ca], [y - s * R_DRONE * sa, y + s * R_DRONE * sa])
        for j, ang in enumerate(np.pi / 4 + np.arange(4) * np.pi / 2):
            rotors[j].center = (x + s * R_DRONE * np.cos(d["psi"][i] + ang), y + s * R_DRONE * np.sin(d["psi"][i] + ang))
            rotors[j].set_facecolor(coul)
        fleche.xy = (x + s * 2.2 * R_DRONE * c, y + s * 2.2 * R_DRONE * si); fleche.set_position((x, y))
        ombre.center = (x, y)
        texte.set_text(f"t = {t[i]:5.2f} s   phase : {PHASES[int(ph[i])]}\n"
                       f"z = {d['z'][i]:.2f} m   écart à la voiture = {eh[i] * 100:.0f} cm\n"
                       f"cap drone {np.degrees(d['psi'][i] - 2 * np.pi):+.0f}°, voiture {np.degrees(cap[i]):+.0f}°")
        pg_v.set_data([cx], [cy]); pg_d.set_data([x], [y]); cur.set_xdata([t[i], t[i]])
        return []

    fig.tight_layout()
    an = animation.FuncAnimation(fig, image, frames=len(idx), interval=1000 * TS * a.pas, blit=False)
    an.save(a.sortie, writer=animation.PillowWriter(fps=round(1 / (TS * a.pas))), dpi=80)
    print(f"{len(idx)} images ({(t[idx[-1]] - t[idx[0]]):.1f} s de vol) -> {a.sortie} "
          f"({Path(a.sortie).stat().st_size / 1e6:.1f} Mo)")


if __name__ == "__main__":
    main()
