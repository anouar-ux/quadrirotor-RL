"""
Trajectoire de la VOITURE pour la mission finale : modèle cinématique (unicycle), vitesse et courbure lisses.

    x' = v cos(cap),  y' = v sin(cap),  cap' = v * kappa

v(s) et kappa(s) (s = temps depuis le démarrage) sont des paliers raccordés par des transitions en S (« smoothstep »,
dérivée nulle aux deux bouts) : accélération et variation de braquage bornées, comme une vraie voiture.
La trajectoire est intégrée UNE FOIS sur la grille du contrôleur (TS = 0,05 s, méthode des trapèzes) et enregistrée en
table : Python et Simulink voient exactement la même voiture (pas de formule à recopier).

Voiture de la mission (VOITURE_MISSION) : garée en (1,5 ; 0,5), elle démarre à t = 4 s (pendant le scan du drone),
accélère à 0,5 m/s, ralentit à 0,35 m/s pour un virage à gauche (rayon 3 m), réaccélère à 0,6 m/s, prend une courbe
douce à droite (rayon 6 m), puis freine à 0,3 m/s. Vitesse max 0,6 m/s, accélération longitudinale <= 0,25 m/s²,
accélération latérale <= 0,06 m/s².
"""
import numpy as np

TS = 0.05

# (instant de début de la transition [s depuis le démarrage], durée [s], nouvelle valeur)
VOITURE_MISSION = dict(
    t_depart=4.0, x0=1.5, y0=0.5, cap0=0.0,
    vitesse=((0.0, 3.0, 0.5), (10.0, 2.0, 0.35), (20.0, 4.0, 0.6), (30.0, 3.0, 0.3)),
    courbure=((12.0, 1.5, 1 / 3), (18.0, 1.5, 0.0), (26.0, 2.0, -1 / 6), (34.0, 2.0, 0.0)),
)


def _smooth(u):
    u = np.clip(u, 0.0, 1.0)
    return u * u * (3 - 2 * u)


def _profil(s, transitions):
    """Valeur d'un profil (paliers + transitions en S) aux instants s, valeur initiale 0."""
    val = np.zeros_like(s)
    prec = 0.0
    for t0, d, v in transitions:
        val += (v - prec) * _smooth((s - t0) / d)
        prec = v
    return val


def trajectoire(p=VOITURE_MISSION, t_fin=60.0):
    """Table de la voiture sur t = 0, TS, ..., t_fin : dict t, x, y, vx, vy, cap, v, kappa (tableaux numpy)."""
    t = np.arange(int(round(t_fin / TS)) + 1) * TS
    s = np.maximum(t - p["t_depart"], 0.0)
    v = np.where(t >= p["t_depart"], _profil(s, p["vitesse"]), 0.0)
    k = _profil(s, p["courbure"])
    w = v * k
    cap = p["cap0"] + np.r_[0.0, np.cumsum(TS / 2 * (w[1:] + w[:-1]))]
    vx, vy = v * np.cos(cap), v * np.sin(cap)
    x = p["x0"] + np.r_[0.0, np.cumsum(TS / 2 * (vx[1:] + vx[:-1]))]
    y = p["y0"] + np.r_[0.0, np.cumsum(TS / 2 * (vy[1:] + vy[:-1]))]
    return dict(t=t, x=x, y=y, vx=vx, vy=vy, cap=cap, v=v, kappa=k)


def voiture_aleatoire(rng):
    """Voiture tirée au hasard (essais de robustesse) : départ, vitesses 0,25-0,7 m/s, 2 virages de rayon >= 2,5 m."""
    tv = np.sort(rng.uniform(0, 30, 4)); tk = np.sort(rng.uniform(6, 36, 4))
    vit = tuple((float(tv[i]), float(rng.uniform(2, 4)), float(rng.uniform(0.25, 0.7))) for i in range(4))
    vit = ((0.0, float(rng.uniform(2, 4)), float(rng.uniform(0.3, 0.6))),) + vit[1:]
    cour = []
    for i in (0, 2):
        cour += [(float(tk[i]), 1.5, float(rng.choice([-1, 1]) * rng.uniform(1 / 8, 1 / 2.5))),
                 (float(max(tk[i + 1], tk[i] + 2)), 1.5, 0.0)]
    return dict(t_depart=float(rng.uniform(2, 6)), x0=float(rng.uniform(0.5, 2.5)), y0=float(rng.uniform(-1.5, 1.5)),
                cap0=float(rng.uniform(-np.pi, np.pi)), vitesse=vit, courbure=tuple(cour))


def lire(table, t):
    """(x, y, vx, vy) de la voiture à l'instant t (indice du pas : pas d'interpolation, comme dans MATLAB)."""
    i = min(int(round(t / TS)), len(table["t"]) - 1)
    return table["x"][i], table["y"][i], table["vx"][i], table["vy"][i]


def lire_cap(table, t):
    """(cap, vitesse de cap v * kappa) de la voiture à l'instant t."""
    i = min(int(round(t / TS)), len(table["t"]) - 1)
    return table["cap"][i], table["v"][i] * table["kappa"][i]


if __name__ == "__main__":
    T = trajectoire()
    a_long = np.diff(T["v"]) / TS
    a_lat = T["v"] ** 2 * np.abs(T["kappa"])
    print(f"vitesse max {T['v'].max():.2f} m/s ; accélération longitudinale max {np.abs(a_long).max():.3f} m/s² ; "
          f"latérale max {a_lat.max():.3f} m/s² ; cap final {np.degrees(T['cap'][-1]):.0f}° ; "
          f"position à 16 s ({T['x'][320]:.2f} ; {T['y'][320]:.2f}), à 60 s ({T['x'][-1]:.2f} ; {T['y'][-1]:.2f})")
