"""
Étape C3 : essais PRBS LONGS de la position (x, y), agent x réutilisé sur les deux axes contre la cascade LQ + PD.

    python python/evaluation/essais_prbs_xy.py        ->  resultats/essais_xy/ (prbs_xy.png, prbs_xy_resultats.json, prbs_xy_trajectoires.npz)

Deux essais, avec le bruit de mesure de g_noise = 0,1 (écart-type 0,0045 sur toutes les mesures, psi compris) :
  A. axe y seul, cap fixe psi = 0 : PRBS 7 bits 0 <-> 1,5 m, Tb = 1,5 s (même essai que « x fin » de essais_prbs.py,
     pour vérifier que l'axe y miroir se comporte comme l'axe x) ;
  B. x ET y en même temps (deux PRBS indépendantes : y 7 bits Tb 1,5 s, x 6 bits Tb 2 s, 0 <-> 1,5 m) pendant que le
     drone TOURNE en continu à 0,2 rad/s (6 tours en 190 s) : chaque consigne passe sans cesse d'un axe du drone à
     l'autre ; c'est l'essai le plus dur pour la rotation par psi.
Indicateurs : RMSE, bande passante à -3 dB (H1 = Pry / Prr, cohérence > 0,8) et dépassements (> 5 cm) par transition.
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # dossier python/ (chemins.py)
from chemins import RESULTATS  # noqa: E402
from env_z import TS  # noqa: E402
from essais_prbs import indicateurs, instants_niveaux  # noqa: E402
from sim_xy import SIGMA_G01, politique_agent, politique_pd, vol  # noqa: E402

SORTIE = RESULTATS / "essais_xy"
T0 = 2.0
W_ROT = 0.2                     # vitesse de rotation du cap dans l'essai B [rad/s]


def evenements(n_bits, tb, haut, n_periodes=1):
    niv, t_fin = instants_niveaux(n_bits, tb, T0, 0.0, haut, n_periodes=n_periodes)
    ev, prec = [], 0.0
    for t, v in niv:
        if v != prec:
            ev.append((t, "p", v - prec)); prec = v
    return tuple(ev), t_fin


def depassements(T, c, ev, seuil=0.05):
    """Nombre de transitions où la position dépasse le niveau visé de plus de seuil (dans le sens du saut)."""
    t, p = T["t"], T[c]
    n, cible = 0, 0.0
    for i, (ti, _, d) in enumerate(ev):
        cible += d
        tf = ev[i + 1][0] if i + 1 < len(ev) else t[-1]
        s = (t >= ti) & (t < tf)
        if s.any() and ((p[s] - cible).max() if d > 0 else (cible - p[s]).max()) > seuil:
            n += 1
    return n, len(ev)


def main():
    SORTIE.mkdir(parents=True, exist_ok=True)
    ev_y, tf_y = evenements(7, 1.5, 1.5)
    ev_x, _ = evenements(6, 2.0, 1.5, n_periodes=2)
    ev_x = tuple(e for e in ev_x if e[0] < tf_y)
    n_pas = int(round((tf_y + 2.0) / TS))
    ctrl = {"agent PPO x": politique_agent(), "LQ + PD 0,3/0,45": politique_pd()}
    essais = {"A": dict(ev_x=(), ev_y=ev_y, psi=0.0, titre="A. axe y seul, ψ = 0"),
              "B": dict(ev_x=ev_x, ev_y=ev_y, psi=lambda t: W_ROT * t,
                        titre=f"B. x et y ensemble, le drone tourne à {W_ROT} rad/s")}
    R, V, H = {}, {}, {}
    for cle, E in essais.items():
        for nom, pol in ctrl.items():
            T = vol(pol, psi_fixe=E["psi"], scan=None, ev_x=E["ev_x"], ev_y=E["ev_y"], sigma=SIGMA_G01, graine=3,
                    n_pas=n_pas)
            V[(cle, nom)] = T
            r = {}
            for c, ev in (("y", E["ev_y"]), ("x", E["ev_x"])):
                if not ev:
                    continue
                ind, h = indicateurs(dict(t=T["t"], t0=T0, ref=T[f"{c}_ref"], y=T[c],
                                          a=T["phi_ref" if c == "y" else "theta_ref"] / 0.2))
                nd, ntr = depassements(T, c, ev)
                ind["depassements_5cm"] = f"{nd}/{ntr}"
                r[c] = ind; H[(cle, nom, c)] = h
            R[f"{cle} | {nom}"] = r
            print(f"{E['titre']:45s} {nom:18s} " + " | ".join(
                f"{c}: RMSE {v['rmse']:.3f} m, bande {v['bande_passante_hz']:.2f} Hz, dépass. {v['depassements_5cm']}"
                for c, v in r.items()))
    with open(SORTIE / "prbs_xy_resultats.json", "w", encoding="utf-8") as f:
        json.dump(R, f, indent=2, ensure_ascii=False)
    np.savez_compressed(SORTIE / "prbs_xy_trajectoires.npz",
                        **{f"{c}_{n.split()[0]}_{k}": v for (c, n), T in V.items() for k, v in T.items()})

    fig, axs = plt.subplots(3, 2, figsize=(14, 10))
    sty = {"agent PPO x": "-", "LQ + PD 0,3/0,45": "--"}
    fen = lambda T: T["t"] < 62
    for j, (cle, c) in enumerate((("A", "y"), ("B", "y"))):
        ax = axs[0, j]
        T = V[(cle, "agent PPO x")]; s = fen(T)
        ax.plot(T["t"][s], T[f"{c}_ref"][s], "k:", label="consigne")
        for nom in ctrl:
            T = V[(cle, nom)]; ax.plot(T["t"][s], T[c][s], sty[nom], lw=1.1, label=nom)
        ax.set_title(f"{essais[cle]['titre']} : {c}(t) (60 premières s)"); ax.set_ylabel(f"{c} [m]"); ax.grid(True)
        ax.legend(fontsize=8)
    ax = axs[1, 0]
    T = V[("B", "agent PPO x")]; s = fen(T)
    ax.plot(T["t"][s], T["x_ref"][s], "k:", label="consigne")
    for nom in ctrl:
        T = V[("B", nom)]; ax.plot(T["t"][s], T["x"][s], sty[nom], lw=1.1, label=nom)
    ax.set_title(f"{essais['B']['titre']} : x(t)"); ax.set_ylabel("x [m]"); ax.grid(True); ax.legend(fontsize=8)
    ax = axs[1, 1]
    T = V[("B", "agent PPO x")]; s = fen(T)
    ax.plot(T["t"][s], np.mod(T["psi"][s], 2 * np.pi), "k", lw=1, label="ψ (mod 2π)")
    ax.plot(T["t"][s], T["theta_ref"][s] * 10, lw=0.9, label="10 θ_ref"); ax.plot(T["t"][s], T["phi_ref"][s] * 10, lw=0.9, label="10 φ_ref")
    ax.set_title("Essai B, agent : cap et consignes d'inclinaison"); ax.set_xlabel("t [s]"); ax.grid(True)
    ax.legend(fontsize=8)
    ax = axs[2, 0]
    for (cle, nom, c), (f, g, coh, _) in H.items():
        ok = coh > 0.8
        ax.semilogx(f[ok], 20 * np.log10(g[ok]), sty[nom], marker=".", ms=3, lw=1,
                    label=f"{cle}, {c}, {nom}")
    ax.axhline(-3, color="k", ls=":"); ax.set_xlabel("f [Hz]"); ax.set_ylabel("|H| [dB]")
    ax.set_title("Réponse en fréquence position / consigne (cohérence > 0,8)"); ax.grid(True, which="both")
    ax.legend(fontsize=7)
    ax = axs[2, 1]
    for cle in essais:
        for nom in ctrl:
            T = V[(cle, nom)]
            e = np.hypot(T["x_ref"] - T["x"], T["y_ref"] - T["y"])
            ax.plot(T["t"], e, sty[nom], lw=0.8, label=f"{cle}, {nom}")
    ax.set_yscale("log"); ax.set_ylim(1e-3, 3); ax.set_xlabel("t [s]"); ax.set_ylabel("|erreur| [m]")
    ax.set_title("Erreur de position sur tout le vol (stabilité sur 190 s)"); ax.grid(True, which="both")
    ax.legend(fontsize=7)
    fig.suptitle("Étape C3 : PRBS longs sur la position (x, y), g_noise = 0,1 (modèle ARX identifié)")
    fig.tight_layout(); fig.savefig(SORTIE / "prbs_xy.png", dpi=115)
    print(f"-> {SORTIE}")


if __name__ == "__main__":
    main()
