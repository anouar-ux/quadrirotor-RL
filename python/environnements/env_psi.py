"""
Environnement Gymnasium du LACET (étape B2) : même structure que env_z.py (z), sur le modèle GRU du lacet.

Physique : u_psi = U_PSI_MAX * a (a dans [-1, 1], action de l'agent) -> GRU -> wdz -> trapèzes -> wz -> psi.
Consigne : une suite de SAUTS DE VITESSE de lacet, chacun lissé par un 2e ordre critique (TAU_W). Entre deux sauts,
           le drone tourne à vitesse constante (scan) ou reste immobile (vitesse 0). Formules fermées en t, sans
           mémoire, à recopier dans le contrôleur MATLAB.
Le z est supposé tenu par l'agent z (aucun couplage mesuré à l'étape B1).
"""
import sys
from pathlib import Path

import gymnasium as gym
import numpy as np
from gymnasium import spaces

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # dossier python/ (chemins.py)
from chemins import MATLAB, PYTHON  # noqa: E402  (rend importables les sous-dossiers de python/)
HERE = PYTHON                # racine python/ : modeles/, resultats/ ; HERE.parent = EI_RL/
from env_z import TS, _t_pas
from identification_psi import LacetGRU

MODEL_PATH = HERE / "modeles" / "psi" / "gru_psi.npz"
N_STEPS = 400              # 20 s de vol
U_PSI_MAX = 0.03           # u_psi = U_PSI_MAX * a : jusqu'à ~5 rad/s² (gain ~172 rad/s² par unité), dans la plage des données
TAU_W = 0.35               # lissage des sauts de vitesse de consigne [s]
W_MAX = 1.5                # vitesses de consigne tirées dans [-W_MAX, W_MAX] rad/s à l'entraînement

# Mission de lacet de la tâche avancée : immobile, puis UN TOUR COMPLET à vitesse constante entre 5 et 13 s
T_SCAN = (5.0, 13.0)
W_SCAN = 2 * np.pi / (T_SCAN[1] - T_SCAN[0])          # ~0.785 rad/s
MISSION_SCAN = ((T_SCAN[0], W_SCAN), (T_SCAN[1], 0.0))  # (instant du saut [s], nouvelle vitesse [rad/s])

# Intégrale de l'erreur (même principe que pour z)
I_MAX = 0.5                # borne [rad.s]
E_INT = 0.1                # intégration conditionnelle si |e| < E_INT [rad] (~6°)

# Récompense : même forme que pour z, échelles en radians
W_FIN, SIG_FIN = 0.5, 0.03       # précision : gros gain sous ~2°
W_LARGE, SIG_LARGE = 0.5, 0.30   # guidage moyen
W_LOIN, E_LOIN = 0.2, 1.0        # guidage lointain (pente constante jusqu'à 1 rad)
W_VIT, SIG_VIT = 0.2, 0.30       # suivi de la vitesse de consigne
LAMBDA_DU = 3.0                  # douceur : - LAMBDA_DU * (a - a_prev)^2


def _S(x):
    """Intégrale de la réponse indicielle lissée : S(x) = x - 2 TAU + (x + 2 TAU) e^(-x/TAU)  (x >= 0)."""
    return x - 2 * TAU_W + (x + 2 * TAU_W) * np.exp(-x / TAU_W)


def consigne_lacet(t, sauts):
    """(psi_ref, w_ref) à l'instant t. sauts = ((t_1, w_1), (t_2, w_2), ...) : la vitesse passe de w_(i-1) à w_i
    à t_i (w_0 = 0), en douceur. psi_ref est l'intégrale exacte de w_ref, psi_ref(0) = 0."""
    t = _t_pas(t)
    psi, w, w_prec = 0.0, 0.0, 0.0
    for ti, wi in sauts:
        if t >= ti:
            x, dw = t - ti, wi - w_prec
            psi += dw * _S(x)
            w += dw * (1.0 - (1.0 + x / TAU_W) * np.exp(-x / TAU_W))
        w_prec = wi
    return float(psi), float(w)


def recompense(e, e_v, du):
    return (W_FIN * np.exp(-(e / SIG_FIN) ** 2) + W_LARGE * np.exp(-(e / SIG_LARGE) ** 2)
            - W_LOIN * min(abs(e), E_LOIN) / E_LOIN - W_VIT * (1.0 - np.exp(-(e_v / SIG_VIT) ** 2)) - LAMBDA_DU * du ** 2)


def observation(m, int_e, w_ref):
    """6 composantes, fonction des seules mesures m (+ intégrale et vitesse de consigne). Même calcul en MATLAB."""
    return np.array([
        (m["psi_ref"] - m["psi"]) / 0.5,   # erreur d'angle
        m["wz"] / 1.0,                     # vitesse de lacet
        m["wdz_prev"] / 5.0,               # accélération de lacet du pas précédent
        m["a_prev"],                       # action précédente (u_psi_prev / U_PSI_MAX)
        int_e / I_MAX,                     # intégrale conditionnelle
        (w_ref - m["wz"]) / 1.0,           # erreur de vitesse
    ], dtype=np.float32)


def maj_integrale(int_e, e):
    if abs(e) < E_INT:
        int_e += e * TS
    return float(np.clip(int_e, -I_MAX, I_MAX))


def mission_aleatoire(rng):
    """1 à 4 sauts de vitesse, au moins 1 s d'écart ; 40 % de retours à l'arrêt, sinon une vitesse dans +-W_MAX."""
    n = int(rng.integers(1, 5))
    t = np.sort(rng.uniform(0.5, N_STEPS * TS - 2.0, n))
    t = t[np.r_[True, np.diff(t) > 1.0]]
    return tuple((float(ti), 0.0 if rng.random() < 0.4 else float(rng.uniform(-W_MAX, W_MAX))) for ti in t)


class YawEnv(gym.Env):
    def __init__(self, sauts_fixes=None, sigma_psi=0.0, sigma_wz=0.0, sigma_wdz=0.0, model_path=MODEL_PATH):
        super().__init__()
        self.model = LacetGRU.load(model_path)
        self.sigma = dict(psi=sigma_psi, wz=sigma_wz, wdz=sigma_wdz)   # bruit de MESURE (0 pour l'export)
        self.sauts_fixes = sauts_fixes        # None : mission aléatoire à chaque vol ; sinon ex. MISSION_SCAN
        self.action_space = spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(6,), dtype=np.float32)

    def _mesures(self):
        t = self.k * TS
        b = lambda s: float(self.np_random.normal(0.0, s)) if s > 0 else 0.0
        return {"t": t,
                "psi": self.psi + b(self.sigma["psi"]),
                "wz": self.wz + b(self.sigma["wz"]),
                "wdz_prev": 0.0 if self.k == 0 else self.wdz_prev + b(self.sigma["wdz"]),
                "a_prev": self.a_prev,
                "psi_ref": consigne_lacet(t, self.sauts)[0]}

    def _obs_info(self):
        m = self._mesures()
        self.int_e = maj_integrale(self.int_e, m["psi_ref"] - m["psi"])
        return observation(m, self.int_e, consigne_lacet(m["t"], self.sauts)[1]), {"m": m}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.k = 0
        self.psi, self.wz = 0.0, 0.0
        self.wdz = self.model.reset()
        self.wdz_prev, self.a_prev, self.int_e = 0.0, 0.0, 0.0
        self.sauts = mission_aleatoire(self.np_random) if self.sauts_fixes is None else tuple(self.sauts_fixes)
        return self._obs_info()

    def step(self, action):
        a = float(np.clip(np.asarray(action).reshape(-1)[0], -1.0, 1.0))
        du = a - self.a_prev
        wdz_next = self.model.step(U_PSI_MAX * a)                 # u_psi(k) -> wdz(k+1)
        wz_next = self.wz + TS / 2 * (self.wdz + wdz_next)
        psi_next = self.psi + TS / 2 * (self.wz + wz_next)
        self.wdz_prev, self.a_prev = self.wdz, a
        self.psi, self.wz, self.wdz = psi_next, wz_next, wdz_next
        self.k += 1
        psi_ref, w_ref = consigne_lacet(self.k * TS, self.sauts)
        reward = recompense(psi_ref - self.psi, w_ref - self.wz, du)   # état VRAI, comme pour z
        obs, info = self._obs_info()
        return obs, float(reward), False, self.k >= N_STEPS, info
