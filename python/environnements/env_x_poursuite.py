"""
Environnement Gymnasium de la POURSUITE sur un axe (variante de env_x.py) : le drone suit la référence que le superviseur
de la mission donne à l'agent de position, et se pose sur une voiture qui roule jusqu'à 2,5 m/s.

Pourquoi un environnement de plus : l'agent de env_x.py a été entraîné avec des cibles lentes (|v| <= 1 m/s, pas d'accélération
soutenue) et reçoit sa vitesse ABSOLUE. Sur une voiture à 2 m/s il sort de son domaine d'entraînement : erreur de traînage,
et le superviseur n'autorise jamais la descente. Ici :

  * la cible est la référence de la phase de poursuite (sim_mission.py) :
        x_ref = x_voiture + D0 (1 + s/tau) exp(-s/tau),   v_ref = v_voiture - D0 s/tau^2 exp(-s/tau)
    (D0 = écart drone - voiture au début ; avec D0 de quelques mètres elle demande PLUS que l'accélération disponible
    g_x * THETA_MAX ~ 2 m/s^2 : le drone doit apprendre à freiner avant d'arriver) ;
  * le drone part parfois avec une vitesse élevée (il « arrive en trombe ») ;
  * l'observation est dans le repère de la référence : vitesse RELATIVE (vx - v_ref) et accélération de la référence
    (filtrée comme dans le contrôleur), à la place de la vitesse absolue.

Observation (7 composantes, comme env_x.py) :
    [ (x_ref - x)/2,  (vx - v_ref)/2,  theta/THETA_MAX,  wy,  a_prev,  integrale/I_MAX,  a_ref_filtree/(g_x THETA_MAX) ]

Récompense : celle de env_x.py (précision, guidage, douceur) avec, pour apprendre à freiner en approche :
  1. suivi d'une COURBE DE FREINAGE à la place du suivi de la vitesse de consigne : à l'écart e de la référence, la vitesse
     relative visée est  v*(e) = sign(e) min(V_APPROCHE, sqrt(2 A_FREIN |e|))  (vitesse d'où l'on peut encore s'arrêter) ;
  2. une pénalité de DÉPASSEMENT quand le drone traverse la référence avec une vitesse relative non nulle ;
  3. un bonus de POSE tant que les conditions de descente du superviseur sont réunies (écart à la voiture < 10 cm et
     vitesse relative < 0,2 m/s).
"""
import sys
from pathlib import Path

import gymnasium as gym
import numpy as np
from gymnasium import spaces

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # dossier python/ (chemins.py)
from chemins import PYTHON  # noqa: E402,F401  (rend importables les sous-dossiers de python/)
from env_z import TS  # noqa: E402
from env_x import (BoucleTangage, charger_boucle_interne, maj_integrale, N_STEPS, THETA_MAX, I_MAX,  # noqa: E402
                   W_FIN, SIG_FIN, W_LARGE, SIG_LARGE, W_LOIN, E_LOIN, W_VIT, SIG_VIT, LAMBDA_DU)

# --- voiture et référence
V_CAR_MAX = 2.5            # vitesse maximale de la voiture à la fin du curriculum [m/s]
A_CAR_MAX = 1.5            # accélération maximale de la voiture dans le repère du drone (virages compris) [m/s^2]
D0_MAX = 4.0               # écart initial drone - voiture [m]
P_PROCHE = 0.2             # proportion des vols qui démarrent presque au-dessus de la voiture
P_V0 = 0.5                 # proportion des vols où le drone démarre avec une vitesse non nulle (arrivée en trombe)
TAU_APP = 1.0              # lissage de l'approche de la mission [s]
P_TAU = 0.4                # proportion des vols avec un tau_app tiré dans [1, TAU_MAX] (robustesse)
TAU_MAX = 2.5
FILT_A = 0.2               # filtre de l'accélération de référence (le même dans le contrôleur) : aF += FILT_A (a - aF)
A_OBS_MAX = 2.0            # l'accélération de référence normalisée est bornée à +-2

# --- récompense
A_FREIN = 1.0              # décélération de la courbe de freinage [m/s^2] : la moitié de g_x THETA_MAX (marge pour la boucle d'attitude)
V_APPROCHE = 3.0           # vitesse d'approche maximale visée [m/s]
W_DEP = 2.0                # pénalité de dépassement : W_DEP min(|vitesse relative|, 2) au pas où l'on traverse la référence
W_POSE = 0.5               # bonus par pas quand les conditions de descente sont réunies
TOL_POSE_X, TOL_POSE_V = 0.10, 0.20


def _smooth(u):
    u = np.clip(u, 0.0, 1.0)
    return u * u * (3 - 2 * u)


def profil_voiture(rng, v_max, evenements=None, v0=None):
    """Vitesse, position de la voiture aux pas k = 0..N_STEPS (x_voiture(0) = 0). Paliers raccordés par des transitions en S
    de durée max(0,8 ; 1,5 |dv| / a_lim) : l'accélération reste <= a_lim. evenements = ((t, v_cible), ...) pour un scénario fixe."""
    t = np.arange(N_STEPS + 1) * TS
    if evenements is None:
        v0 = float(rng.uniform(-v_max, v_max)) if rng.random() < 0.7 else 0.0
        a_lim = float(rng.uniform(0.3, A_CAR_MAX))
        ts = np.sort(rng.uniform(0.5, 16.0, int(rng.integers(1, 5))))
        evenements = [(float(ti), 0.0 if rng.random() < 0.25 else float(rng.uniform(-v_max, v_max))) for ti in ts]
    else:
        a_lim = A_CAR_MAX
    v, prec, libre = np.full_like(t, v0 or 0.0), (v0 or 0.0), 0.0
    for ti, vt in evenements:
        if ti < libre:
            continue                                          # transition précédente pas finie : on saute l'événement
        d = max(0.8, 1.5 * abs(vt - prec) / a_lim)
        v += (vt - prec) * _smooth((t - ti) / d)
        prec, libre = vt, ti + d
    x = np.r_[0.0, np.cumsum(TS / 2 * (v[1:] + v[:-1]))]
    return v, x


def scenario(v_car, x_car, D0, tau=TAU_APP, v_drone0=0.0):
    """Scénario complet : voiture, référence lissée (formules de sim_mission.py) et état initial du drone."""
    s = np.arange(N_STEPS + 1) * TS
    e = np.exp(-s / tau)
    return dict(v_car=v_car, x_car=x_car, D0=float(D0), tau=float(tau), v_drone0=float(v_drone0),
                x_ref=x_car + D0 * (1 + s / tau) * e, v_ref=v_car - D0 * s / tau ** 2 * e)


def scenario_aleatoire(rng, v_max=V_CAR_MAX):
    v, x = profil_voiture(rng, v_max)
    D0 = float(rng.uniform(-D0_MAX, D0_MAX)) * (0.1 if rng.random() < P_PROCHE else 1.0)
    tau = float(rng.uniform(TAU_APP, TAU_MAX)) if rng.random() < P_TAU else TAU_APP
    v_d0 = float(rng.uniform(-V_CAR_MAX, V_CAR_MAX)) if rng.random() < P_V0 else 0.0
    return scenario(v, x, D0, tau, v_d0)


def observation_poursuite(m, g_x):
    """7 composantes, fonction des seules mesures m. Même calcul dans le contrôleur de la mission."""
    return np.array([
        (m["x_ref"] - m["x"]) / 2.0,                                  # erreur de position
        (m["vx"] - m["v_ref"]) / 2.0,                                 # vitesse relative à la référence
        m["theta"] / THETA_MAX,                                       # angle de tangage mesuré
        m["wy"] / 1.0,                                                # vitesse de tangage
        m["a_prev"],                                                  # action précédente
        m["int_e"] / I_MAX,                                           # intégrale conditionnelle
        np.clip(m["a_ref"] / (g_x * THETA_MAX), -A_OBS_MAX, A_OBS_MAX),   # accélération de la référence, en fraction de l'accélération disponible
    ], dtype=np.float32)


def recompense_poursuite(e, e_v, du, courbe=True):
    """e = x_ref - x, e_v = vx - v_ref (état vrai). courbe=False : suivi de la vitesse de consigne comme env_x.recompense."""
    if courbe:
        e_v = e_v - np.sign(e) * min(V_APPROCHE, np.sqrt(2 * A_FREIN * abs(e)))
    return (W_FIN * np.exp(-(e / SIG_FIN) ** 2) + W_LARGE * np.exp(-(e / SIG_LARGE) ** 2)
            - W_LOIN * min(abs(e), E_LOIN) / E_LOIN - W_VIT * (1.0 - np.exp(-(e_v / SIG_VIT) ** 2)) - LAMBDA_DU * du ** 2)


class PoursuiteEnv(gym.Env):
    def __init__(self, scenario_fixe=None, sigma_x=0.0, sigma_vx=0.0, sigma_theta=0.0, sigma_wy=0.0, v_max=1.0, courbe=True):
        super().__init__()
        self.boucle = BoucleTangage(charger_boucle_interne())
        self.g_x = self.boucle.p["g_x"]
        self.sigma = dict(x=sigma_x, vx=sigma_vx, theta=sigma_theta, wy=sigma_wy, wdy=sigma_wy)
        self.scenario_fixe, self.v_max, self.courbe = scenario_fixe, v_max, courbe
        self.action_space = spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(7,), dtype=np.float32)

    def set_vmax(self, v_max):
        """Curriculum : vitesse maximale de la voiture des PROCHAINS vols."""
        self.v_max = float(v_max)

    def _mesures(self):
        k, B = self.k, self.boucle
        b = lambda s: float(self.np_random.normal(0.0, s)) if s > 0 else 0.0
        self._bruit = (b(self.sigma["theta"]), b(self.sigma["wy"]), b(self.sigma["wdy"]))
        sc = self.sc
        return {"t": k * TS, "x": B.x + b(self.sigma["x"]), "vx": B.vx + b(self.sigma["vx"]),
                "theta": B.theta + self._bruit[0], "wy": B.wy + self._bruit[1], "a_prev": self.a_prev,
                "x_ref": sc["x_ref"][k], "v_ref": sc["v_ref"][k], "a_ref": self.aF}

    def _obs_info(self):
        m = self._mesures()
        self.int_e = maj_integrale(self.int_e, m["x_ref"] - m["x"])
        m["int_e"] = self.int_e
        return observation_poursuite(m, self.g_x), {"m": m}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.sc = self.scenario_fixe if self.scenario_fixe is not None else scenario_aleatoire(self.np_random, self.v_max)
        self.k = 0
        B = self.boucle
        B.reset()
        B.x, B.vx = self.sc["x_car"][0] + self.sc["D0"], self.sc["v_drone0"]     # le drone démarre à D0 de la voiture
        B.ax = B.p["frot_x"] * B.vx
        self.a_prev, self.int_e, self.aF = 0.0, 0.0, 0.0
        self.e_prev = 0.0
        return self._obs_info()

    def step(self, action):
        a = float(np.clip(np.asarray(action).reshape(-1)[0], -1.0, 1.0))
        return self.avance(THETA_MAX * a, a)

    def avance(self, theta_ref, a):
        """Un pas avec une consigne d'angle theta_ref (rad) ; a = action de l'agent mémorisée pour l'observation suivante
        (a = theta_ref / THETA_MAX tant qu'aucune anticipation n'est ajoutée)."""
        du = a - self.a_prev
        self.boucle.step(float(np.clip(theta_ref, -0.3, 0.3)), 0.0, self._bruit)
        self.a_prev = a
        self.k += 1
        k, B, sc = self.k, self.boucle, self.sc
        # accélération de référence : différence finie de v_ref puis filtre, comme le contrôleur
        self.aF += FILT_A * ((sc["v_ref"][k] - sc["v_ref"][k - 1]) / TS - self.aF)
        e, e_v = sc["x_ref"][k] - B.x, B.vx - sc["v_ref"][k]                     # état VRAI
        r = recompense_poursuite(e, e_v, du, self.courbe)
        if self.e_prev * e < 0 and abs(e_v) > 0.1:                                  # traversée de la référence en vitesse
            r -= W_DEP * min(abs(e_v), 2.0)
        if abs(B.x - sc["x_car"][k]) < TOL_POSE_X and abs(B.vx - sc["v_car"][k]) < TOL_POSE_V:
            r += W_POSE
        self.e_prev = e
        obs, info = self._obs_info()
        return obs, float(r), False, self.k >= N_STEPS, info
