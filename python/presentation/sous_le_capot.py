"""
SOUS LE CAPOT : comment le drone « se penche » pour suivre une voiture qui démarre, freine et recule (GIF).

Vol des agents RL de presentation/suivi_voiture.py (conduite « ligne droite », de 1,5 à 12 s : démarrage sec, coup de
frein, marche arrière, arrêt). La voiture roule selon x et le cap du drone reste nul : l'avant du drone est l'axe x.
Panneaux :
  - vue de dessus : voiture, drone et sa flèche d'accélération horizontale ;
  - drone vu de côté (tangage theta) et vu de derrière (roulis phi) : silhouette inclinée, poussée de chaque moteur
    (flèche proportionnelle à la commande), silhouette pointillée = inclinaison DEMANDÉE par l'agent x (theta_ref) ;
  - les 4 commandes moteur (barres ; tiret : vol stationnaire u = 0,39) ;
  - au cours du temps : vitesse de la voiture et du drone, theta demandé par l'agent et theta obtenu.
Position des moteurs déduite du mélangeur (u_theta = (u3 - u1) / 2, u_phi = (u2 - u4) / 2) et des signes identifiés
(a_x = +9,9 theta, a_y = -9,9 phi) : M1 avant, M3 arrière, M2 gauche, M4 droite.

    python python/presentation/sous_le_capot.py     ->  rapport/presentation/07_sous_le_capot.gif
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import animation  # noqa: E402

import commun as C  # noqa: E402
import sim_mission as M  # noqa: E402

TS = C.TS
T0, T1, PAS = 1.5, 12.0, 2
L_BRAS = 1.0
EXAG = 2.0              # inclinaison DESSINÉE = 2 x l'inclinaison réelle (quelques degrés : sinon invisible)
GAIN_FL = 10.0          # flèche de poussée : longueur 0,9 + 10 x (u - u_stationnaire)


def silhouette(ax, titre, gauche, droite):
    ax.set_xlim(-1.9, 1.9); ax.set_ylim(-0.9, 1.9); ax.set_aspect("equal"); ax.axis("off")
    ax.set_title(titre, fontsize=10)
    ax.axhline(-0.75, color="0.6", lw=1)
    corps, = ax.plot([], [], color="k", lw=5, solid_capstyle="round", zorder=3)
    consigne, = ax.plot([], [], color="tab:blue", lw=2, ls="--", zorder=2)
    rot = [ax.plot([], [], color="0.3", lw=3, zorder=4)[0] for _ in range(2)]
    fl = [ax.annotate("", xy=(0, 0), xytext=(0, 0), zorder=5, arrowprops=dict(arrowstyle="-|>", lw=2.5, color="tab:orange"))
          for _ in range(2)]
    noms = [ax.text(0, 0, n, fontsize=8, ha="center", va="top", zorder=6) for n in (gauche, droite)]
    angle = ax.text(0, 1.75, "", ha="center", va="top", fontsize=10, family="monospace")
    tir = ax.annotate("", xy=(0, 0), xytext=(0, 0), zorder=1, arrowprops=dict(arrowstyle="-|>", lw=1.5, color="tab:red", alpha=0.6))

    def maj(ang, ang_ref, u_g, u_d, accel, nom_ang):
        """ang > 0 : poussée inclinée vers la droite de l'écran (sens de l'accélération)."""
        vrai, vrai_ref = ang, ang_ref
        ang, ang_ref = EXAG * ang, EXAG * ang_ref
        c, s = np.cos(ang), np.sin(ang)
        bout =lambda sg, a: (sg * L_BRAS * np.cos(a), -sg * L_BRAS * np.sin(a))
        g, d = bout(-1, ang), bout(1, ang)
        corps.set_data([g[0], d[0]], [g[1], d[1]])
        gr, dr = bout(-1, ang_ref), bout(1, ang_ref)
        consigne.set_data([gr[0], dr[0]], [gr[1], dr[1]])
        for k, (p, u) in enumerate(((g, u_g), (d, u_d))):
            rot[k].set_data([p[0] - 0.22 * c, p[0] + 0.22 * c], [p[1] + 0.22 * s, p[1] - 0.22 * s])
            n = (s, c)                                             # normale au corps = direction de la poussée
            Lp = max(0.9 + GAIN_FL * (u - M.U_EQ), 0.05)
            fl[k].set_position((p[0], p[1])); fl[k].xy = (p[0] + Lp * n[0], p[1] + Lp * n[1])
            noms[k].set_position((p[0], p[1] - 0.12))
        tir.set_position((0, 0)); tir.xy = (0.35 * accel, 0)
        angle.set_text(f"{nom_ang} = {np.degrees(vrai):+5.1f}°  (demandé {np.degrees(vrai_ref):+5.1f}°)")
    return maj


def main():
    C.style_clair()
    titre, param, duree = C.CONDUITES["ligne_droite"]
    T = C.table_voiture(param, duree + C.T_CHAUFFE)
    R = C.vol_famille("RL", T, duree, suivi=True)
    t, vr, Y, vt = R["t"], R["vrai"], R["y"], R["voiture"]
    i0, i1 = int(round(T0 / TS)), int(round(T1 / TS))
    idx = np.arange(i0, i1, PAS)
    th, ph, th_r, ph_r = vr[:, 7], vr[:, 6], Y[:, 11], Y[:, 12]
    u = Y[:, :4]
    ax_d = vr[:, 12]                                              # accélération horizontale du drone (x)
    # contrôle des signes du dessin : u3 - u1 > 0 doit faire monter theta (accélération angulaire wdy > 0, avec le retard
    # des moteurs : ~3 pas), et theta > 0 doit accélérer le drone vers l'avant (a_x > 0)
    c1 = max(np.corrcoef(u[:len(u) - k, 2] - u[:len(u) - k, 0], vr[k:, 16])[0, 1] for k in range(5))
    c2 = np.corrcoef(th, ax_d)[0, 1]
    print(f"contrôle des signes : corr(u3 - u1, wdy) = {c1:+.2f}, corr(theta, a_x) = {c2:+.2f} (attendus > 0)")

    fig = plt.figure(figsize=(12, 6.8))
    fig.suptitle("Sous le capot : le drone s'incline pour accélérer et freiner avec la voiture (agents RL)", fontsize=12)
    axv = fig.add_axes((0.03, 0.42, 0.27, 0.48))
    axc = fig.add_axes((0.32, 0.50, 0.3, 0.40)); axf = fig.add_axes((0.32, 0.08, 0.3, 0.40))
    axb = fig.add_axes((0.70, 0.55, 0.27, 0.33))
    axt = fig.add_axes((0.70, 0.08, 0.27, 0.18)); axa = fig.add_axes((0.70, 0.30, 0.27, 0.18))
    axl = fig.add_axes((0.03, 0.08, 0.25, 0.25)); axl.axis("off")
    # vue de dessus
    axv.set_aspect("equal"); axv.grid(True, alpha=0.5); axv.set_title("vue de dessus", fontsize=10); axv.tick_params(labelsize=7)
    voit, dr = C.DessinVoiture(axv), C.DessinDrone(axv)
    fl_a = axv.annotate("", xy=(0, 0), xytext=(0, 0), zorder=9, arrowprops=dict(arrowstyle="-|>", lw=2.5, color="tab:red"))
    # silhouettes
    maj_c = silhouette(axc, "vu de côté : tangage θ (avant à droite)", "M3 arrière", "M1 avant")
    maj_f = silhouette(axf, "vu de derrière : roulis φ (droite du drone à droite)", "M2 gauche", "M4 droite")
    # barres moteurs
    barres = axb.bar(["M1\navant", "M2\ngauche", "M3\narrière", "M4\ndroite"], [0] * 4, color="tab:orange", ec="k")
    axb.axhline(0, color="k", lw=1)
    axb.set_ylim(-0.07, 0.07); axb.set_title("commande de chaque moteur\n(écart au vol stationnaire u = 0,39)", fontsize=9)
    axb.tick_params(labelsize=8); axb.grid(axis="y", alpha=0.4)
    # courbes
    sl = slice(i0, i1)
    axa.plot(t[sl], vt["v"][sl], color="0.3", lw=2, label="voiture")
    axa.plot(t[sl], vr[sl, 3], color="tab:blue", lw=1.2, label="drone")
    axa.set_ylabel("vitesse [m/s]", fontsize=8); axa.legend(fontsize=7, loc="lower left", ncol=2); axa.grid(alpha=0.4)
    axa.tick_params(labelsize=7); axa.set_xticklabels([])
    axt.plot(t[sl], np.degrees(th_r[sl]), color="tab:blue", ls="--", lw=1.2, label="θ demandé par l'agent")
    axt.plot(t[sl], np.degrees(th[sl]), color="k", lw=1.2, label="θ obtenu")
    axt.set_ylabel("θ [°]", fontsize=8); axt.set_xlabel("t [s]", fontsize=8); axt.legend(fontsize=7, loc="lower left", ncol=2)
    axt.grid(alpha=0.4); axt.tick_params(labelsize=7)
    curs = [a.axvline(t[i0], color="k", lw=1) for a in (axa, axt)]
    legende = axl.text(0, 1, "", va="top", fontsize=9.5, family="monospace", transform=axl.transAxes)
    bandeau = fig.text(0.165, 0.385, "", ha="center", va="top", fontsize=14, weight="bold", color="tab:orange")
    fig.text(0.47, 0.02, f"inclinaisons dessinées x {EXAG:.0f} ; pointillés bleus : inclinaison demandée par l'agent x ; "
             "flèches orange : poussée des moteurs ; flèche rouge : accélération horizontale", ha="center", fontsize=8, color="0.3")
    import suivi_voiture as S
    kv = np.minimum(np.round((t + C.T_CHAUFFE) / TS).astype(int), len(T["t"]) - 1)
    Tk = {c: T[c][kv] for c in ("a_long", "v", "kappa")}

    def image(j):
        i = idx[j]
        cx, cy = vt["x"][i], vt["y"][i]
        axv.set_xlim(cx - 0.7, cx + 0.7); axv.set_ylim(cy - 0.7, cy + 0.7)
        voit.maj(cx, cy, vt["cap"][i]); dr.maj(vr[i, 0], vr[i, 1], vr[i, 8])
        fl_a.set_position((vr[i, 0], vr[i, 1])); fl_a.xy = (vr[i, 0] + 0.25 * vr[i, 12], vr[i, 1] + 0.25 * vr[i, 13])
        maj_c(th[i], th_r[i], u[i, 2], u[i, 0], vr[i, 12], "θ")
        maj_f(ph[i], ph_r[i], u[i, 1], u[i, 3], -vr[i, 13], "φ")
        for b, v in zip(barres, u[i]):
            b.set_height(v - M.U_EQ); b.set_color("tab:red" if v > M.U_EQ + 0.01 else ("tab:cyan" if v < M.U_EQ - 0.01 else "0.7"))
            b.set_edgecolor("k")
        for c in curs:
            c.set_xdata([t[i], t[i]])
        legende.set_text(f"t = {t[i]:5.2f} s\nvoiture  {vt['v'][i]:+.2f} m/s\n         {Tk['a_long'][i]:+.1f} m/s²\n"
                         f"drone    {vr[i, 3]:+.2f} m/s\n         {vr[i, 12]:+.1f} m/s²\nécart    {100 * R['eh'][i]:.0f} cm")
        bandeau.set_text(S.evenement(Tk, i))
        return []

    an = animation.FuncAnimation(fig, image, frames=len(idx), interval=1000 * TS * PAS, blit=False)
    C.enregistrer_gif(an, C.PRESENTATION / "07_sous_le_capot.gif", fps=round(1 / (TS * PAS)), dpi=72)


if __name__ == "__main__":
    main()
