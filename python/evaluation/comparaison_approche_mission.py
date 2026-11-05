"""
Mission finale : influence de la consigne d'approche (TAU_APP) sur la comparaison agent x / PD, et option de cap.

    python python/evaluation/comparaison_approche_mission.py      ->  resultats/mission/comparaison_approche.{json,png}

Constat de départ : avec TAU_APP = 1 s, la consigne d'approche elle-même n'arrive à 5 cm de la voiture que ~7 s après
le début de la poursuite ; le PD, qui dépasse sa consigne de ~12 cm, entre plus tôt dans la fenêtre d'alignement.
On rejoue la mission (g_noise 0,1) sur la voiture nominale et 12 voitures aléatoires pour TAU_APP = 1 ; 0,5 ; 0,25 s,
avec l'agent x et avec le PD, cap fixe ; puis la meilleure approche avec le cap orienté vers la voiture.
Indicateurs (état VRAI) : durée de poursuite (début poursuite -> début descente), temps pour passer une première fois
à moins de 20 cm de la voiture, puis, à partir de cet instant : écart max à la voiture et dépassement (drone DEVANT la
voiture, le long de sa vitesse) ; écart au contact.
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # dossier python/ (chemins.py)
import chemins  # noqa: E402,F401  (rend importables les sous-dossiers de python/)
import sim_mission as M  # noqa: E402
import voiture as V  # noqa: E402

TAUS = (1.0, 0.5, 0.25)
N_ALEA = 12


def indicateurs(L, tab):
    d = M.bilan_complet(L, tab)
    t, ph, vr, y = L["v"][:, 0], L["y"][:, 4], L["vrai"], L["y"]
    i = (ph >= 3) & (ph < 5)
    eh = np.hypot(vr[:, 0] - y[:, 9], vr[:, 1] - y[:, 10])
    proche = np.where(i & (eh < 0.20))[0]
    apres = np.zeros_like(i)
    if len(proche):
        apres[proche[0]:] = True
        apres &= i
    idx = np.round(t / M.TS).astype(int)
    vx, vy = tab["vx"][idx], tab["vy"][idx]
    n = np.maximum(np.hypot(vx, vy), 1e-6)
    devant = ((vr[:, 0] - y[:, 9]) * vx + (vr[:, 1] - y[:, 10]) * vy) / n
    d["poursuite_s"] = (d["t_descente"] - d["t_poursuite"]) if d["t_descente"] else None
    d["rattrape_20cm_s"] = float(t[proche[0]] - d["t_poursuite"]) if len(proche) else None
    d["ecart_apres_rattrapage_m"] = float(eh[apres].max()) if apres.any() else None
    d["depasse_voiture_m"] = float(max(devant[apres].max(), 0.0)) if apres.any() else None
    d["sature_s"] = float(np.sum(i & ((np.abs(vr[:, 7]) > 0.19) | (np.abs(vr[:, 6]) > 0.19))) * M.TS)
    return d


def resume(liste):
    ok = [d for d in liste if d["t_posé"] is not None]
    f = lambda k, g=np.mean: float(g([d[k] for d in ok])) if ok else float("nan")
    return dict(poses=f"{len(ok)}/{len(liste)}", poursuite_moy_s=f("poursuite_s"), poursuite_max_s=f("poursuite_s", max),
                rattrape_moy_s=f("rattrape_20cm_s"), ecart_apres_max_m=f("ecart_apres_rattrapage_m", max),
                depasse_max_m=f("depasse_voiture_m", max), depasse_moy_m=f("depasse_voiture_m"),
                sature_moy_s=f("sature_s"), contact_max_m=f("ecart_contact_m", max),
                v_rel_max_ms=f("vitesse_relative_horiz_ms", max))


def main():
    P = M.charger_politiques()
    tabs = [V.trajectoire(t_fin=61.0)] + [V.trajectoire(V.voiture_aleatoire(np.random.default_rng(100 + g)), t_fin=61.0)
                                          for g in range(N_ALEA)]
    R, nominal = {}, {}
    cas = [(tau, mode, "fixe") for tau in TAUS for mode in ("rl", "pd")]
    cas += [(0.5, "rl", "voiture"), (0.5, "pd", "voiture")]
    for tau, mode, cap in cas:
        cle = f"tau={tau} | {'agent' if mode == 'rl' else 'PD'} | cap {cap}"
        L = [indicateurs(M.vol(P, tab, mode, M.SIGMA_G01, graine=g, tau_app=tau, cap=cap), tab)
             for g, tab in enumerate(tabs)]
        R[cle] = resume(L); nominal[cle] = L[0]
        r = R[cle]
        print(f"{cle:34s} posé {r['poses']:6s} poursuite {r['poursuite_moy_s']:5.1f} s (max {r['poursuite_max_s']:5.1f}) | "
              f"à 20 cm en {r['rattrape_moy_s']:4.1f} s | après : écart max {r['ecart_apres_max_m']*100:4.1f} cm, devant max {r['depasse_max_m']*100:4.1f} cm | "
              f"contact max {r['contact_max_m']*100:4.1f} cm | v rel max {r['v_rel_max_ms']*100:4.1f} cm/s")
    with open(M.SORTIE / "comparaison_approche.json", "w", encoding="utf-8") as f:
        json.dump(dict(resume_13_voitures=R, voiture_nominale=nominal), f, indent=2, ensure_ascii=False)

    fig, axs = plt.subplots(1, 3, figsize=(15, 4.5))
    for j, (k, lab) in enumerate((("poursuite_moy_s", "durée de poursuite moyenne [s]"),
                                  ("ecart_apres_max_m", "écart max après rattrapage (< 20 cm) [m]"),
                                  ("contact_max_m", "écart au contact max [m]"))):
        ax = axs[j]
        for mode, c in (("agent", "C0"), ("PD", "C1")):
            ax.plot(TAUS, [R[f"tau={t} | {mode} | cap fixe"][k] for t in TAUS], "o-", color=c, label=f"{mode}, cap fixe")
            ax.plot([0.5], [R[f"tau=0.5 | {mode} | cap voiture"][k]], "*", ms=12, color=c, label=f"{mode}, cap vers la voiture")
        ax.set_xlabel("τ de l'approche [s]"); ax.set_ylabel(lab); ax.grid(True); ax.invert_xaxis()
        if j == 0:
            ax.legend(fontsize=8)
    fig.suptitle(f"Mission finale, g_noise = 0,1, voiture nominale + {N_ALEA} voitures aléatoires : effet de l'approche")
    fig.tight_layout(); fig.savefig(M.SORTIE / "comparaison_approche.png", dpi=115)
    print(f"-> {M.SORTIE}")


if __name__ == "__main__":
    main()
