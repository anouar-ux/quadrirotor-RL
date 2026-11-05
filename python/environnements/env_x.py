"""
Environnement Gymnasium de la POSITION x (étape C2) : l'agent donne une consigne d'angle theta_ref, le LQ d'attitude
de l'étape C1 la suit (boucle interne), et le drone avance.

Physique (tout est identifié, étape C1) :
    theta_ref = THETA_MAX * a                         (a dans [-1, 1], action de l'agent)
    u(k)      = -K [theta - theta_ref, wy, wdy(k), wdy(k-1), u(k-1..k-5)]     LQ sur le modèle de gains_LQ_ARX_attitude (C1),
                                                                               réglage plus rapide (Q_ANG, Q_W, R_ATT)
    wdy(k+1)  = a1 wdy(k) + a2 wdy(k-1) + b1 u(k) + ... + b6 u(k-5)           ARX(2,6) identifié (moyenne theta/phi)
    wy, theta : trapèzes ;  ax = G_X * theta + FROT_X * vx (translation identifiée) ;  vx, x : trapèzes
Consigne : sauts de POSITION et sauts de VITESSE (cible qui roule), chacun lissé par un 2e ordre critique (TAU_X).
Formules fermées en t, sans mémoire, à recopier dans le contrôleur MATLAB.
z et psi sont supposés tenus par leurs agents (z : anticipation en 1/cos(theta) ajoutée dans le contrôleur).

Mesure de theta : depuis la correction du bloc de bruit de UAV_model (Cov = 0.0001 comme les autres sorties,
2026-11-03), theta est MESURÉ DIRECTEMENT (bruit blanc ~0.0045 rad à g_noise 0.1 : sigma_theta). L'option derive
(marche aléatoire sur l'angle vu par le LQ) ne sert plus que si l'on revient à theta_est = intégrale de wy ;
elle reste disponible comme test de robustesse (biais d'angle lent : capteur mal calé, masse décentrée).
"""
import sys
import json
from pathlib import Path

import gymnasium as gym
import numpy as np
import scipy.io as sio
from gymnasium import spaces
from scipy.linalg import solve_discrete_are

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # dossier python/ (chemins.py)
from chemins import COMMANDE_LQ, MATLAB, PYTHON  # noqa: E402  (rend importables les sous-dossiers de python/)
HERE = PYTHON                # racine python/ : modeles/, resultats/ ; HERE.parent = EI_RL/
from env_z import TS, _t_pas

LQ_PATH = COMMANDE_LQ / "LQ_ARX_attitude" / "gains_LQ_ARX_attitude.mat"
RESULTS_ATT = HERE / "modeles" / "attitude" / "results.json"

N_STEPS = 400              # 20 s de vol

# LQ d'attitude de la boucle interne, recalculé sur le modèle ARX de C1 (voir charger_boucle_interne) :
# moins de pénalité sur la vitesse angulaire et sur la commande que C1 -> 3 fois plus rapide.
Q_ANG, Q_W, R_ATT = 100.0, 1.0, 200.0
THETA_MAX = 0.2            # theta_ref = THETA_MAX * a [rad] : même borne que la cascade PD (accélération max ~2 m/s²)
U_MAX_ATT = 0.5            # borne de la commande différentielle de tangage (marge des moteurs autour de u_z ~ 0.39)
TAU_X = 1.0                # lissage des sauts de consigne [s] : un saut de 2 m demande au plus ~2 m/s², faisable

# Mission aléatoire d'entraînement
DX_MAX = 2.0               # amplitude des sauts de position [m]
X_LIM = 4.0                # la cible de position reste dans [-X_LIM, X_LIM] (hors vitesse)
V_MAX = 1.0                # vitesses de cible tirées dans [-V_MAX, V_MAX] [m/s]
P_VITESSE = 0.3            # proportion des événements qui sont des sauts de vitesse (cible qui roule)

# Dérive de l'estimation theta_est = intégrale de wy (mesurée sur le vol Simulink de C1 à g_noise 0.1 :
# ~0.003 rad en 6 s). Modèle : marche aléatoire, écart-type SIG_DERIVE * sqrt(t). Tirée par vol dans [0, SIG_DERIVE_MAX].
SIG_DERIVE_MAX = 0.002     # [rad/sqrt(s)]

# Intégrale de l'erreur (même principe que z et psi)
I_MAX = 0.5                # borne [m.s]
E_INT = 0.1                # intégration conditionnelle si |e| < E_INT [m]

# Récompense : même forme que z et psi, échelles en mètres
W_FIN, SIG_FIN = 0.5, 0.05       # précision : gros gain sous ~5 cm
W_LARGE, SIG_LARGE = 0.5, 0.50   # guidage moyen
W_LOIN, E_LOIN = 0.2, 2.0        # guidage lointain (pente constante jusqu'à 2 m)
W_VIT, SIG_VIT = 0.2, 0.30       # suivi de la vitesse de consigne
LAMBDA_DU = 3.0                  # douceur : - LAMBDA_DU * (a - a_prev)^2


def charger_boucle_interne(reglage="rapide", lq_path=LQ_PATH, results_path=RESULTS_ATT):
    """ARX moyen (a, b) et modèle d'état (A, B) de l'étape C1, gain LQ K (9 termes, sans intégrale), translation.
    reglage = "C1"     : K de gains_LQ_ARX_attitude.mat (q 100/10, R 2e4 ; montée 0,85 s) -> TROP LENT en boucle interne :
                         la cascade PD x (0,23 / 0,28) est instable (bande passante voisine de celle de x) ;
    reglage = "rapide" : K recalculé ici sur le MÊME modèle avec Q_ANG, Q_W, R_ATT (montée 0,30 s, 2,8 %)."""
    d = sio.loadmat(lq_path, squeeze_me=True)
    with open(results_path, encoding="utf-8") as f:
        tr = json.load(f)["translation_x"]
    if reglage == "C1":
        K = np.asarray(d["K"], float)[:9]
    else:
        A, B = np.asarray(d["A"], float)[:9, :9], np.asarray(d["B"], float)[:9].reshape(-1, 1)
        Q = np.diag([Q_ANG, Q_W] + [0.0] * 7)
        P = solve_discrete_are(A, B, Q, np.array([[R_ATT]]))
        K = np.linalg.solve(R_ATT + B.T @ P @ B, B.T @ P @ A).ravel()
    return dict(K=K, a=np.asarray(d["a"], float), b=np.asarray(d["b"], float),
                g_x=float(tr["gain_angle"]), frot_x=float(tr["frottement"]))


def _lisse(s):
    """Réponse indicielle lissée (2e ordre critique) et sa dérivée, pour s >= 0."""
    e = np.exp(-s / TAU_X)
    return 1.0 - (1.0 + s / TAU_X) * e, s / TAU_X ** 2 * e


def _S(s):
    """Intégrale de la réponse indicielle lissée : S(s) = s - 2 TAU + (s + 2 TAU) e^(-s/TAU)."""
    return s - 2 * TAU_X + (s + 2 * TAU_X) * np.exp(-s / TAU_X)


def consigne_x(t, evenements):
    """(x_ref, v_ref) à l'instant t. evenements = ((t_i, 'p', dx), (t_j, 'v', v), ...) :
    'p' : saut de position de dx [m] ; 'v' : la vitesse de la cible passe à v [m/s] (elle roule).
    Chaque événement est lissé par un 2e ordre critique de constante TAU_X ; x_ref(0) = 0, v_ref(0) = 0."""
    t = _t_pas(t)
    x, v, v_prec = 0.0, 0.0, 0.0
    for ti, typ, val in evenements:
        if t >= ti:
            s = t - ti
            r, dr = _lisse(s)
            if typ == "p":
                x += val * r
                v += val * dr
            else:
                dv = val - v_prec
                x += dv * _S(s)
                v += dv * r
        if typ == "v":
            v_prec = val
    return float(x), float(v)


def recompense(e, e_v, du):
    return (W_FIN * np.exp(-(e / SIG_FIN) ** 2) + W_LARGE * np.exp(-(e / SIG_LARGE) ** 2)
            - W_LOIN * min(abs(e), E_LOIN) / E_LOIN - W_VIT * (1.0 - np.exp(-(e_v / SIG_VIT) ** 2)) - LAMBDA_DU * du ** 2)


def observation(m, int_e, v_ref):
    """7 composantes, fonction des seules mesures m (+ intégrale et vitesse de consigne). Même calcul en MATLAB."""
    return np.array([
        (m["x_ref"] - m["x"]) / 2.0,      # erreur de position
        m["vx"] / 1.0,                    # vitesse
        m["theta"] / THETA_MAX,           # angle de tangage mesuré
        m["wy"] / 1.0,                    # vitesse de tangage
        m["a_prev"],                      # action précédente (theta_ref_prev / THETA_MAX)
        int_e / I_MAX,                    # intégrale conditionnelle
        (v_ref - m["vx"]) / 1.0,          # erreur de vitesse
    ], dtype=np.float32)


def maj_integrale(int_e, e):
    if abs(e) < E_INT:
        int_e += e * TS
    return float(np.clip(int_e, -I_MAX, I_MAX))


def mission_aleatoire(rng):
    """1 à 3 événements, au moins 2 s d'écart. 70 % : saut de position de +-DX_MAX (cible gardée dans +-X_LIM) ;
    30 % : la cible se met à rouler à une vitesse dans +-V_MAX (ou s'arrête, 1 fois sur 3)."""
    n = int(rng.integers(1, 4))
    t = np.sort(rng.uniform(0.5, N_STEPS * TS - 4.0, n))
    t = t[np.r_[True, np.diff(t) > 2.0]]
    ev, xc = [], 0.0
    for ti in t:
        if rng.random() < P_VITESSE:
            v = 0.0 if rng.random() < 1 / 3 else float(rng.uniform(-V_MAX, V_MAX))
            ev.append((float(ti), "v", v))
        else:
            dx = float(np.clip(xc + rng.uniform(-DX_MAX, DX_MAX), -X_LIM, X_LIM) - xc)
            xc += dx
            ev.append((float(ti), "p", dx))
    return tuple(ev)


class BoucleTangage:
    """Boucle interne : LQ d'attitude (C1) + ARX du tangage + translation en x. Un pas = TS."""

    def __init__(self, p):
        self.p = p
        self.reset()

    def reset(self):
        self.theta, self.wy, self.wdy, self.wdy_prev = 0.0, 0.0, 0.0, 0.0
        self.n_wdy_prev = 0.0                 # bruit de la mesure wdy du pas précédent (mémorisée par le LQ)
        self.U = np.zeros(5)                  # u(k-1..k-5), commandes APPLIQUÉES (après saturation)
        self.x, self.vx = 0.0, 0.0
        self.ax = 0.0

    def step(self, theta_ref, biais=0.0, bruit=(0.0, 0.0, 0.0)):
        """biais : erreur lente de l'angle vu par le LQ (option derive). Le LQ tient theta + biais = theta_ref.
        bruit : bruit de mesure (theta, wy, wdy) vu par le LQ ce pas-ci, comme sur Simulink."""
        p = self.p
        n_th, n_wy, n_wdy = bruit
        etat = np.r_[self.theta + biais + n_th - theta_ref, self.wy + n_wy, self.wdy + n_wdy, self.wdy_prev + self.n_wdy_prev,
                     self.U]
        self.n_wdy_prev = n_wdy
        u = float(np.clip(-p["K"] @ etat, -U_MAX_ATT, U_MAX_ATT))
        wdy_next = p["a"][0] * self.wdy + p["a"][1] * self.wdy_prev + p["b"][0] * u + p["b"][1:] @ self.U
        wy_next = self.wy + TS / 2 * (self.wdy + wdy_next)
        theta_next = self.theta + TS / 2 * (self.wy + wy_next)
        # translation : ax = G_X theta + FROT vx (implicite en vx_next, résolu exactement)
        g, f = p["g_x"], p["frot_x"]
        vx_next = (self.vx + TS / 2 * (self.ax + g * theta_next)) / (1.0 - TS / 2 * f)
        ax_next = g * theta_next + f * vx_next
        x_next = self.x + TS / 2 * (self.vx + vx_next)
        self.U = np.r_[u, self.U[:4]]
        self.wdy_prev, self.wdy, self.wy, self.theta = self.wdy, wdy_next, wy_next, theta_next
        self.vx, self.ax, self.x = vx_next, ax_next, x_next
        return u


class XEnv(gym.Env):
    def __init__(self, evenements_fixes=None, sigma_x=0.0, sigma_vx=0.0, sigma_theta=0.0, sigma_wy=0.0, derive=False):
        """derive : False -> aucune dérive de theta_est ; True -> SIG_DERIVE tiré dans [0, SIG_DERIVE_MAX] à chaque vol ;
        un nombre -> SIG_DERIVE imposé."""
        super().__init__()
        self.boucle = BoucleTangage(charger_boucle_interne())
        # bruit de MESURE (0 pour l'export) ; wdy (vu par le LQ seulement) a le même écart-type que wy
        self.sigma = dict(x=sigma_x, vx=sigma_vx, theta=sigma_theta, wy=sigma_wy, wdy=sigma_wy)
        self.derive = derive
        self.evenements_fixes = evenements_fixes   # None : mission aléatoire ; sinon ex. ((1.0, 'p', 2.0),)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(7,), dtype=np.float32)

    def _mesures(self):
        t = self.k * TS
        b = lambda s: float(self.np_random.normal(0.0, s)) if s > 0 else 0.0
        B = self.boucle
        # mêmes mesures pour l'agent et pour le LQ au pas k (comme dans le bloc contrôleur de Simulink)
        self._bruit = (b(self.sigma["theta"]), b(self.sigma["wy"]), b(self.sigma["wdy"]))
        return {"t": t, "x": B.x + b(self.sigma["x"]), "vx": B.vx + b(self.sigma["vx"]),
                "theta": B.theta + self.biais + self._bruit[0], "wy": B.wy + self._bruit[1],
                "a_prev": self.a_prev, "x_ref": consigne_x(t, self.ev)[0]}

    def _obs_info(self):
        m = self._mesures()
        self.int_e = maj_integrale(self.int_e, m["x_ref"] - m["x"])
        return observation(m, self.int_e, consigne_x(m["t"], self.ev)[1]), {"m": m}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.k = 0
        self.boucle.reset()
        self.a_prev, self.int_e, self.biais = 0.0, 0.0, 0.0
        if self.derive is True:
            self.sig_derive = float(self.np_random.uniform(0.0, SIG_DERIVE_MAX))
        else:
            self.sig_derive = float(self.derive or 0.0)
        self.ev = mission_aleatoire(self.np_random) if self.evenements_fixes is None else tuple(self.evenements_fixes)
        return self._obs_info()

    def step(self, action):
        a = float(np.clip(np.asarray(action).reshape(-1)[0], -1.0, 1.0))
        du = a - self.a_prev
        self.boucle.step(THETA_MAX * a, self.biais, self._bruit)
        if self.sig_derive > 0:
            self.biais += float(self.np_random.normal(0.0, self.sig_derive * np.sqrt(TS)))
        self.a_prev = a
        self.k += 1
        x_ref, v_ref = consigne_x(self.k * TS, self.ev)
        reward = recompense(x_ref - self.boucle.x, v_ref - self.boucle.vx, du)   # état VRAI, comme pour z et psi
        obs, info = self._obs_info()
        return obs, float(reward), False, self.k >= N_STEPS, info
