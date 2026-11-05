"""
SUIVI D'UNE VOITURE À FORTS À-COUPS (sans atterrissage) : GIF pour la présentation.

Le drone part au-dessus de la voiture, à 0,5 m, et doit y RESTER pendant que la voiture conduit brutalement :
  - « ligne droite » : démarrages secs, coups de frein, marche arrière (accélérations de pointe ~2 m/s², soit la
    limite du drone : theta_max = 0,2 rad -> 9,9 x 0,2 = 2 m/s²) ;
  - « slalom » : virages de rayon 1,2 m enchaînés, demi-tour de rayon 0,9 m.
Simulateur assemblé (modèles identifiés), mesures bruitées comme g_noise = 0,1, 20 Hz.

    python python/presentation/suivi_voiture.py            # les deux conduites, GIF de suivi (agents) ET duels
    python python/presentation/suivi_voiture.py --rapide   # une image sur 4 (essai de mise en page)

Sorties (rapport/presentation/) :
  01_suivi_ligne_droite.gif, 02_suivi_slalom.gif   agents RL seuls : vue qui suit la voiture, vitesse de la voiture
                                                   (avec les à-coups signalés), écart drone - voiture en cm ;
  03_duel_ligne_droite.gif, 04_duel_slalom.gif     les TROIS familles de contrôleurs sur la même voiture (même bruit) :
                                                   RL (bleu), LQ sur ARX (vert), LQI double intégrateur réajusté (rouge) ;
  05_duel_bilan.png                                écart RMS et maximal par famille et par conduite ;
  suivi_voiture.json                               indicateurs chiffrés.
"""
import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import animation  # noqa: E402

import commun as C  # noqa: E402

TS = C.TS
DEMI_VUE = 0.9          # demi-largeur de la vue qui suit la voiture [m]
FENETRE = 3.0           # traces des 3 dernières secondes


def evenement(T, i):
    """Texte de l'à-coup en cours (pour le bandeau)."""
    a, v, k = T["a_long"][i], T["v"][i], T["kappa"][i] if "kappa" in T else 0.0
    if a < -0.7:
        return "COUP DE FREIN !" if v >= 0 else "FREIN (marche arrière) !"
    if a > 0.7:
        return "DÉMARRAGE SEC !" if v >= 0 else "MARCHE ARRIÈRE !"
    if abs(k) > 0.6 and v > 0.3:
        return "VIRAGE SERRÉ !"
    return ""


def indicateurs(R):
    eh = R["eh"]
    return dict(rms_cm=float(100 * np.sqrt(np.mean(eh ** 2))), max_cm=float(100 * eh.max()),
                part_sous_10cm=float(np.mean(eh < 0.10)), z_min=float(R["vrai"][:, 2].min()), z_max=float(R["vrai"][:, 2].max()),
                theta_max_deg=float(np.degrees(np.abs(R["vrai"][:, 6:8]).max())))


def preparer_vue(ax):
    ax.set_aspect("equal"); ax.grid(True, alpha=0.5)
    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")


def gif(nom, titre, T, vols, sortie, pas):
    """GIF de suivi : vols = {famille: R} (une famille ; plusieurs : drones superposés dans la même vue)."""
    duel = len(vols) > 1
    R0 = next(iter(vols.values()))
    t, vt = R0["t"], R0["voiture"]
    n = len(t)
    kv = np.minimum(np.round((t + C.T_CHAUFFE) / TS).astype(int), len(T["t"]) - 1)
    Tk = {c: T[c][kv] for c in ("a_long", "v", "kappa")}
    idx = np.arange(0, n, pas)
    nf = int(FENETRE / TS)

    fig = plt.figure(figsize=(11.5, 6.2))
    gs = fig.add_gridspec(3, 2, width_ratios=(1.15, 1), height_ratios=(1.1, 0.8, 1))
    axv = fig.add_subplot(gs[:, 0]); axg = fig.add_subplot(gs[0, 1]); axs = fig.add_subplot(gs[1, 1]); axe = fig.add_subplot(gs[2, 1])
    fig.suptitle(("Duel des contrôleurs : " if duel else "Le drone reste au-dessus de la voiture : ") + titre, fontsize=12)
    # vue d'ensemble
    axg.plot(vt["x"], vt["y"], color="0.6", lw=4, alpha=0.5, label="voiture")
    for f, R in vols.items():
        axg.plot(R["vrai"][:, 0], R["vrai"][:, 1], color=C.COUL[f], lw=0.9, label=C.NOMS[f] if duel else "drone")
    pt_v, = axg.plot([], [], "s", color="0.3", ms=6)
    axg.set_aspect("equal", adjustable="datalim"); axg.grid(True, alpha=0.5)
    axg.legend(fontsize=7, loc="best"); axg.set_title("vue d'ensemble", fontsize=9); axg.tick_params(labelsize=7)
    # vitesse de la voiture, à-coups surlignés
    axs.plot(t, vt["v"], color="0.3", lw=1.4, label="vitesse de la voiture [m/s]")
    axs.fill_between(t, -1.2, 1.2, where=np.abs(Tk["a_long"]) > 0.7, color="tab:orange", alpha=0.25, lw=0, label="à-coup (|a| > 0,7 m/s²)")
    if np.abs(Tk["kappa"]).max() > 0:
        axs.fill_between(t, -1.2, 1.2, where=(np.abs(Tk["kappa"]) > 0.6) & (Tk["v"] > 0.3), color="tab:purple", alpha=0.18, lw=0,
                         label="virage serré (rayon < 1,7 m)")
    axs.set_ylim(min(-0.1, vt["v"].min() - 0.1), vt["v"].max() + 0.15); axs.grid(True, alpha=0.5)
    axs.legend(fontsize=7, loc="upper right", ncol=1); axs.tick_params(labelsize=7); axs.set_xlim(t[0], t[-1])
    cur_s = axs.axvline(0, color="k", lw=1)
    # écart horizontal
    lignes = {f: axe.plot([], [], color=C.COUL[f], lw=1.4, label=f"{C.NOMS[f]}")[0] for f in vols}
    axe.axhline(10, color="0.4", ls=":", lw=0.8)
    axe.set_xlim(t[0], t[-1]); axe.set_ylim(0, max(100 * R["eh"].max() for R in vols.values()) * 1.1)
    axe.set_xlabel("t [s]", fontsize=8); axe.set_ylabel("écart drone - voiture [cm]", fontsize=8); axe.grid(True, alpha=0.5)
    axe.tick_params(labelsize=7); axe.legend(fontsize=7, loc="upper left")
    cur_e = axe.axvline(0, color="k", lw=1)
    # vue qui suit la voiture
    preparer_vue(axv)
    trace_v, = axv.plot([], [], color="0.6", lw=5, alpha=0.4, solid_capstyle="round")
    traces = {f: axv.plot([], [], color=C.COUL[f], lw=1.3, alpha=0.9)[0] for f in vols}
    voit = C.DessinVoiture(axv)
    drones = {f: C.DessinDrone(axv, C.COUL[f], echelle=0.8 if duel else 1.0) for f in vols}
    texte = axv.text(0.02, 0.98, "", transform=axv.transAxes, va="top", fontsize=9, family="monospace",
                     bbox=dict(fc="w", ec="0.7", alpha=0.92), zorder=10)
    bandeau = axv.text(0.5, 0.04, "", transform=axv.transAxes, ha="center", va="bottom", fontsize=15, weight="bold",
                       color="tab:orange", zorder=10, bbox=dict(fc="w", ec="none", alpha=0.75))

    def image(j):
        i = idx[j]
        cx, cy = vt["x"][i], vt["y"][i]
        axv.set_xlim(cx - DEMI_VUE, cx + DEMI_VUE); axv.set_ylim(cy - DEMI_VUE, cy + DEMI_VUE)
        a = max(0, i - nf)
        trace_v.set_data(vt["x"][a:i + 1], vt["y"][a:i + 1])
        voit.maj(cx, cy, vt["cap"][i])
        lignes_txt = [f"t = {t[i]:5.1f} s   voiture : {vt['v'][i]:+.2f} m/s, {Tk['a_long'][i]:+.1f} m/s²"]
        for f, R in vols.items():
            x, y, psi = R["vrai"][i, 0], R["vrai"][i, 1], R["vrai"][i, 8]
            traces[f].set_data(R["vrai"][a:i + 1, 0], R["vrai"][a:i + 1, 1])
            drones[f].maj(x, y, psi)
            lignes[f].set_data(t[:i + 1], 100 * R["eh"][:i + 1])
            lignes_txt.append(f"{f:3s} écart {100 * R['eh'][i]:4.0f} cm   (RMS jusqu'ici {100 * np.sqrt(np.mean(R['eh'][:i + 1] ** 2)):4.1f})"
                              if duel else f"écart drone - voiture : {100 * R['eh'][i]:3.0f} cm   inclinaison "
                              f"{np.degrees(np.hypot(R['vrai'][i, 6], R['vrai'][i, 7])):4.1f}°")
        texte.set_text("\n".join(lignes_txt))
        bandeau.set_text(evenement(Tk, i))
        pt_v.set_data([cx], [cy]); cur_s.set_xdata([t[i], t[i]]); cur_e.set_xdata([t[i], t[i]])
        return []

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    an = animation.FuncAnimation(fig, image, frames=len(idx), interval=1000 * TS * pas, blit=False)
    C.enregistrer_gif(an, sortie, fps=max(1, round(1 / (TS * pas))))
    plt.close(fig)


def gif_duel(nom, titre, T, vols, sortie, pas):
    """Trois vues côte à côte (une par famille, chacune centrée sur la voiture), écarts et vitesse de la voiture dessous."""
    R0 = vols["RL"]
    t, vt = R0["t"], R0["voiture"]
    kv = np.minimum(np.round((t + C.T_CHAUFFE) / TS).astype(int), len(T["t"]) - 1)
    Tk = {c: T[c][kv] for c in ("a_long", "v", "kappa")}
    idx = np.arange(0, len(t), pas)
    nf = int(FENETRE / TS)
    fig = plt.figure(figsize=(12, 7.2))                        # positions fixes (aspect égal : pas de tight_layout)
    fig.suptitle("Duel des contrôleurs : " + titre + "  (même voiture, même bruit de mesure)", fontsize=12)
    vues = {}
    for j, f in enumerate(vols):
        ax = fig.add_axes((0.045 + 0.325 * j, 0.40, 0.27, 0.45)); preparer_vue(ax); ax.tick_params(labelsize=7)
        ax.set_title(C.NOMS[f], color=C.COUL[f], fontsize=11, weight="bold")
        ax.set_xlabel("")
        for sp in ax.spines.values():
            sp.set_edgecolor(C.COUL[f]); sp.set_linewidth(2)
        tv, = ax.plot([], [], color="0.6", lw=5, alpha=0.4, solid_capstyle="round")
        tr, = ax.plot([], [], color=C.COUL[f], lw=1.3)
        txt = ax.text(0.5, 0.03, "", transform=ax.transAxes, ha="center", va="bottom", fontsize=10, family="monospace",
                      bbox=dict(fc="w", ec="0.7", alpha=0.92), zorder=10)
        vues[f] = (ax, tv, tr, C.DessinVoiture(ax), C.DessinDrone(ax, C.COUL[f]), txt)
    axe = fig.add_axes((0.06, 0.08, 0.57, 0.24)); axs = fig.add_axes((0.72, 0.08, 0.26, 0.24))
    lignes = {f: axe.plot([], [], color=C.COUL[f], lw=1.5, label=C.NOMS[f])[0] for f in vols}
    axe.axhline(10, color="0.4", ls=":", lw=0.8)
    axe.set_xlim(t[0], t[-1]); axe.set_ylim(0, max(100 * R["eh"].max() for R in vols.values()) * 1.1)
    axe.set_xlabel("t [s]"); axe.set_ylabel("écart drone - voiture [cm]"); axe.grid(True, alpha=0.5)
    axe.legend(fontsize=8, loc="upper left")
    axe.fill_between(t, 0, 200, where=np.abs(Tk["a_long"]) > 0.7, color="tab:orange", alpha=0.2, lw=0)
    if np.abs(Tk["kappa"]).max() > 0:
        axe.fill_between(t, 0, 200, where=(np.abs(Tk["kappa"]) > 0.6) & (Tk["v"] > 0.3), color="tab:purple", alpha=0.12, lw=0)
    cur_e = axe.axvline(0, color="k", lw=1)
    axs.plot(t, vt["v"], color="0.3", lw=1.4)
    axs.fill_between(t, -1.2, 1.2, where=np.abs(Tk["a_long"]) > 0.7, color="tab:orange", alpha=0.25, lw=0, label="à-coup")
    if np.abs(Tk["kappa"]).max() > 0:
        axs.fill_between(t, -1.2, 1.2, where=(np.abs(Tk["kappa"]) > 0.6) & (Tk["v"] > 0.3), color="tab:purple", alpha=0.18,
                         lw=0, label="virage serré")
    axs.set_ylim(min(-0.1, vt["v"].min() - 0.1), vt["v"].max() + 0.15); axs.set_xlim(t[0], t[-1]); axs.grid(True, alpha=0.5)
    axs.set_title("vitesse de la voiture [m/s]", fontsize=9); axs.set_xlabel("t [s]"); axs.legend(fontsize=7, loc="upper right")
    cur_s = axs.axvline(0, color="k", lw=1)
    bandeau = fig.text(0.5, 0.935, "", ha="center", va="top", fontsize=14, weight="bold", color="tab:orange")
    horloge = fig.text(0.015, 0.935, "", ha="left", va="top", fontsize=12, family="monospace")

    def image(j):
        i = idx[j]
        cx, cy = vt["x"][i], vt["y"][i]
        a = max(0, i - nf)
        for f, (ax, tv, tr, voit, dr, txt) in vues.items():
            R = vols[f]
            ax.set_xlim(cx - DEMI_VUE, cx + DEMI_VUE); ax.set_ylim(cy - DEMI_VUE, cy + DEMI_VUE)
            tv.set_data(vt["x"][a:i + 1], vt["y"][a:i + 1]); voit.maj(cx, cy, vt["cap"][i])
            tr.set_data(R["vrai"][a:i + 1, 0], R["vrai"][a:i + 1, 1]); dr.maj(R["vrai"][i, 0], R["vrai"][i, 1], R["vrai"][i, 8])
            txt.set_text(f"écart {100 * R['eh'][i]:3.0f} cm | RMS {100 * np.sqrt(np.mean(R['eh'][:i + 1] ** 2)):4.1f}")
            lignes[f].set_data(t[:i + 1], 100 * R["eh"][:i + 1])
        cur_e.set_xdata([t[i], t[i]]); cur_s.set_xdata([t[i], t[i]])
        bandeau.set_text(evenement(Tk, i)); horloge.set_text(f"t = {t[i]:4.1f} s")
        return []

    an = animation.FuncAnimation(fig, image, frames=len(idx), interval=1000 * TS * pas, blit=False)
    C.enregistrer_gif(an, sortie, fps=max(1, round(1 / (TS * pas))), dpi=72)
    plt.close(fig)


def figure_bilan(IND, fichier):
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    conduites = list(IND)
    larg = 0.26
    for j, (cle, lab) in enumerate((("rms_cm", "écart RMS [cm]"), ("max_cm", "écart maximal [cm]"))):
        for k, f in enumerate(C.FAMILLES):
            vals = [IND[c][f][cle] for c in conduites]
            b = ax[j].bar(np.arange(len(conduites)) + (k - 1) * larg, vals, larg, color=C.COUL[f], label=C.NOMS[f])
            ax[j].bar_label(b, fmt="%.0f" if cle == "max_cm" else "%.1f", fontsize=8)
        ax[j].set_xticks(np.arange(len(conduites)), [C.CONDUITES[c][0].split(" :")[0].replace("slalom serré puis demi-tour", "slalom + demi-tour")
                                                     for c in conduites])
        ax[j].set_ylabel(lab); ax[j].grid(axis="y", alpha=0.4)
    ax[0].legend(fontsize=8)
    fig.suptitle("Suivi d'une voiture à forts à-coups (drone à 0,5 m, sans atterrissage) : écart drone - voiture", fontsize=11)
    fig.tight_layout(); fig.savefig(fichier, dpi=130); plt.close(fig)
    print(f"-> {fichier}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--rapide", action="store_true", help="une image sur 4 (essai de mise en page)")
    a = p.parse_args()
    C.style_clair()
    pas = 4 if a.rapide else 2                                    # 2 : 10 images/s, temps réel
    IND = {}
    for num, (nom, (titre, param, duree)) in enumerate(C.CONDUITES.items()):
        T = C.table_voiture(param, duree + C.T_CHAUFFE)
        vols = {f: C.vol_famille(f, T, duree, suivi=True) for f in C.FAMILLES}
        IND[nom] = {f: indicateurs(R) for f, R in vols.items()}
        for f in C.FAMILLES:
            print(f"{nom:12s} {f:3s} : écart RMS {IND[nom][f]['rms_cm']:5.1f} cm, max {IND[nom][f]['max_cm']:5.1f} cm")
        gif(nom, titre, T, {"RL": vols["RL"]}, C.PRESENTATION / f"0{num + 1}_suivi_{nom}.gif", pas)
        gif_duel(nom, titre, T, vols, C.PRESENTATION / f"0{num + 3}_duel_{nom}.gif", pas)
    figure_bilan(IND, C.PRESENTATION / "05_duel_bilan.png")
    (C.PRESENTATION / "suivi_voiture.json").write_text(json.dumps(IND, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
