"""
CIBLE D'ATTERRISSAGE : où le drone se pose-t-il sur la plateforme de la voiture ? (figure de la présentation)

Mission finale complète (décollage, scan, poursuite, alignement, descente, posé : superviseur de sim_mission.py)
sur les 13 voitures du rapport (voiture de la mission + 12 voitures tirées au hasard, graines 100 à 111, comme dans
sim_mission.main), mesures bruitées (g_noise = 0,1), pour les TROIS familles de contrôleurs (presentation/commun.py).
Le point de posé est exprimé dans le repère de la VOITURE (avant / gauche) au premier instant où z < 2 cm.

    python python/presentation/cible_atterrissage.py     ->  rapport/presentation/06_cible_atterrissage.png
                                                             rapport/presentation/cible_atterrissage.json
    python python/presentation/cible_atterrissage.py --figure   # redessine seulement (à partir du .json)
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import patches  # noqa: E402

import commun as C  # noqa: E402
import sim_mission as M  # noqa: E402
import voiture as V  # noqa: E402

MARQ = {"RL": "o", "ARX": "s", "LQI": "^"}


def voitures():
    yield "voiture de la mission", V.VOITURE_MISSION, 1
    for g in range(12):
        yield f"aléatoire {100 + g}", V.voiture_aleatoire(np.random.default_rng(100 + g)), g


def pose(R):
    """Point de posé dans le repère de la voiture [m], instant et vitesse verticale du contact (None si pas posé)."""
    ph = R["y"][:, 4]
    if not (ph >= 5).any():
        return None
    i = int(np.argmax(ph >= 5))
    d = R["vrai"][i, :2] - np.array([R["voiture"]["x"][i], R["voiture"]["y"][i]])
    c, s = np.cos(R["voiture"]["cap"][i]), np.sin(R["voiture"]["cap"][i])
    return dict(avant_m=float(c * d[0] + s * d[1]), gauche_m=float(-s * d[0] + c * d[1]), t_pose_s=float(R["t"][i]),
                vz_contact_ms=float(R["vrai"][i, 5]), ecart_cap_deg=float(np.degrees(M.wrap(R["vrai"][i, 8] - R["voiture"]["cap"][i]))))


def main():
    C.style_clair()
    res = {f: [] for f in C.FAMILLES}
    for nom, p, graine in voitures():
        T = C.table_voiture(p, 61.0)
        for f in C.FAMILLES:
            d = pose(C.vol_famille(f, T, 60.0, graine=graine))
            res[f].append(dict(voiture=nom, **(d or {"pose": False})))
            print(f"{nom:22s} {f:3s} : " + (f"posé à t = {d['t_pose_s']:.1f} s, écart {100 * np.hypot(d['avant_m'], d['gauche_m']):.1f} cm"
                                             if d else "PAS POSÉ en 60 s"))
    (C.PRESENTATION / "cible_atterrissage.json").write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")

    figure(res)


def figure(res=None):
    """Figure seule, à partir de cible_atterrissage.json (python cible_atterrissage.py --figure)."""
    C.style_clair()
    if res is None:
        res = json.loads((C.PRESENTATION / "cible_atterrissage.json").read_text(encoding="utf-8"))
    fig = plt.figure(figsize=(13.5, 5.6))
    ax = fig.add_axes((0.05, 0.13, 0.30, 0.62)); axz = fig.add_axes((0.385, 0.13, 0.21, 0.62))
    axt = fig.add_axes((0.72, 0.13, 0.27, 0.62))
    ecarts = {f: [100 * np.hypot(d["avant_m"], d["gauche_m"]) for d in res[f] if "avant_m" in d] for f in C.FAMILLES}
    for a, zoom in ((ax, False), (axz, True)):
        # la voiture vue de dessus, à l'échelle, en cm
        a.add_patch(patches.FancyBboxPatch((-22.5, -11), 45, 22, boxstyle="round,pad=0,rounding_size=3", fc="0.85", ec="0.4", zorder=0))
        a.add_patch(patches.Rectangle((-10, -8), 20, 16, fc="w", ec="k", lw=1.2, zorder=1))
        a.add_patch(patches.Rectangle((20, -9), 2.5, 18, fc="gold", ec="none", zorder=1))
        for r, ls in ((1, ":"), (2, "-"), (5, "--")):
            if zoom or r > 1:
                a.add_patch(patches.Circle((0, 0), r, fill=False, ec="0.35", ls=ls, lw=0.9, zorder=2))
                a.text(r * 0.71 + 0.15, r * 0.71 + 0.15, f"{r} cm", fontsize=7, color="0.35")
        for f in C.FAMILLES:
            P = [d for d in res[f] if "avant_m" in d]
            a.scatter([100 * d["avant_m"] for d in P], [100 * d["gauche_m"] for d in P], marker=MARQ[f], s=60 if zoom else 30,
                      c=C.COUL[f], ec="k", lw=0.5, alpha=0.85, zorder=5,
                      label=f"{C.NOMS[f]} : {len(P)}/13 posés, écart moyen {np.mean(ecarts[f]):.1f} cm, max {np.max(ecarts[f]):.1f} cm")
        a.set_aspect("equal")
    ax.set_xlim(-25, 32); ax.set_ylim(-14, 14)
    ax.annotate("", xy=(31, 0), xytext=(23.5, 0), arrowprops=dict(arrowstyle="-|>", color="0.3", lw=1.5))
    ax.text(27, 1.5, "sens de\nla voiture", fontsize=7, ha="center", color="0.3")
    ax.add_patch(patches.Rectangle((-6, -6), 12, 12, fill=False, ec="tab:purple", lw=1, ls="--", zorder=6))
    ax.set_xlabel("vers l'avant de la voiture [cm]"); ax.set_ylabel("vers la gauche [cm]")
    ax.set_title("voiture entière (plateforme blanche : 20 x 16 cm)", fontsize=10)
    axz.set_xlim(-6, 6); axz.set_ylim(-6, 6); axz.grid(alpha=0.3)
    for sp in axz.spines.values():
        sp.set_edgecolor("tab:purple"); sp.set_linestyle("--")
    axz.set_title("zoom sur le centre (12 x 12 cm)", fontsize=10); axz.set_xlabel("vers l'avant [cm]")
    h, l = ax.get_legend_handles_labels()
    fig.legend(h, l, loc="upper left", bbox_to_anchor=(0.05, 0.92), fontsize=8.5, frameon=False, ncol=1)
    # instant du posé, voiture par voiture
    noms = [d["voiture"] for d in res["RL"]]
    for k, f in enumerate(C.FAMILLES):
        tp = [d.get("t_pose_s", np.nan) for d in res[f]]
        axt.barh(np.arange(len(noms)) + (k - 1) * 0.27, tp, 0.27, color=C.COUL[f])
        axt.axvline(np.nanmean(tp), color=C.COUL[f], ls="--", lw=1.2)
        axt.text(0.02, 0.985 - 0.045 * k, f"moyenne {f} : {np.nanmean(tp):.1f} s", color=C.COUL[f], transform=axt.transAxes,
                 fontsize=8, weight="bold", va="top", bbox=dict(fc="w", ec="none", alpha=0.85, pad=0.5))
    axt.set_yticks(np.arange(len(noms)), noms, fontsize=7); axt.invert_yaxis(); axt.set_xlim(0, 52); axt.set_ylim(len(noms) - 0.5, -2.6)
    axt.set_xlabel("instant du posé [s] (tirets : moyenne)"); axt.grid(axis="x", alpha=0.4)
    axt.set_title("instant du posé, voiture par voiture", fontsize=10)
    fig.suptitle("Mission finale sur 13 voitures (mesures bruitées, g_noise = 0,1) : où et quand le drone se pose", fontsize=12, y=0.985)
    fig.savefig(C.PRESENTATION / "06_cible_atterrissage.png", dpi=140); plt.close(fig)
    print(f"-> {C.PRESENTATION / '06_cible_atterrissage.png'}")


if __name__ == "__main__":
    import sys
    figure() if "--figure" in sys.argv else main()
