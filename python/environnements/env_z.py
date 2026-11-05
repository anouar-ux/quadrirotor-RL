"""
ENVIRONNEMENT GYMNASIUM DE L'ALTITUDE z (piste RL, étapes 3 à 5 du sujet).

À quoi sert ce fichier : simuler le drone selon z avec le modèle GRU identifié (identification/identification_z.py,
modeles/z/gru_z.npz) pour entraîner l'agent (entrainement/train_ppo_z.py, train_sac_z.py) :
    action u dans [-1, 1] (commande commune des 4 moteurs)  ->  GRU : az(k+1)  ->  vz, z par la méthode des trapèzes.
Contenu : la consigne lissée (sauts de hauteur filtrés par un 2e ordre critique, formule exacte en t, recopiée dans
rl_controller.m), l'observation à 6 composantes calculée sur les seules MESURES (comme dans Simulink), l'intégrale
conditionnelle de l'erreur, la récompense bornée (précision, guidage, vitesse, douceur), les missions de la Figure 1
(MISSION_FIG1) et de la tâche avancée (MISSION_AVANCEE_Z), et le bruit de mesure optionnel.
"""
import sys
from pathlib import Path

import gymnasium as gym
import numpy as np
from gymnasium import spaces

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # dossier python/ (chemins.py)
from chemins import PYTHON  # noqa: E402  (rend importables les sous-dossiers de python/)
HERE = PYTHON                # racine python/ : modeles/, resultats/ ; HERE.parent = EI_RL/
from identification_z import DroneGRU

MODEL_PATH = HERE / "modeles" / "z" / "gru_z.npz"
TS = 0.05                  # période d'échantillonnage [s]
N_STEPS = 200              # 10 s de vol
Z_REF_RANGE = (-0.5, 1.0)  # consignes tirées au hasard [m]
TAU_REF = 0.35             # constante de temps de la consigne lissée (2e ordre critique) [s]

# Intégrale de l'erreur (seule mémoire du contrôleur, à reproduire dans rl_controller.m)
I_MAX = 0.5                # borne de l'intégrale [m.s] (anti-windup)
E_INT = 0.1                # intégration CONDITIONNELLE : on n'intègre que si |e| < E_INT [m] (régime établi)

# Récompense (version 2) : termes bornés, entre ~ -0.5 et +1 par pas
W_FIN, SIG_FIN = 0.5, 0.05       # suivi fin : gros gain sous ~5 cm (précision, pas d'erreur statique)
W_LARGE, SIG_LARGE = 0.5, 0.30   # suivi large : guide le drone quand il est loin de la consigne
W_LOIN, E_LOIN = 0.2, 1.0        # guidage lointain : pente constante jusqu'à 1 m d'écart (les exponentielles sont plates au-delà)
W_VIT, SIG_VIT = 0.2, 0.30       # suivi de la vitesse de consigne : amortit, limite dépassement et oscillations
LAMBDA_DU = 3.0                  # douceur de la commande : - LAMBDA_DU * (u - u_prev)^2


MISSION_FIG1 = ((0.0, 0.0), (3.0, 0.5), (7.0, -0.25))   # Figure 1 : (instant du saut [s], hauteur [m]) ; instants choisis par le groupe
MISSION_AVANCEE_Z = ((0.0, 0.0), (2.0, 0.5), (15.0, -0.25))   # tâche avancée (20 s) : 0,5 m tenu pendant le scan de 5 à 13 s


def _t_pas(t):
    """t ramené au numéro de pas arrondi : t = 1.9999999 (Simulink) et 2.0 (Python) donnent la même valeur."""
    return round(t / TS) * TS


def _paliers(z_cible):
    """Une hauteur seule (entraînement) = un saut de 0 à z_cible à t = 0 ; sinon une liste de (instant, hauteur)."""
    if np.isscalar(z_cible):
        return ((0.0, 0.0), (0.0, float(z_cible)))
    return tuple((float(ti), float(hi)) for ti, hi in z_cible)


def mission(t, z_cible):
    """Consigne LISSÉE à l'instant t : somme de sauts, chacun lissé par un 2e ordre critique de constante TAU_REF,
        z_ref(t) = h_0 + somme_i (h_i - h_(i-1)) * (1 - (1 + s_i/TAU_REF) * exp(-s_i/TAU_REF)),  s_i = t - t_i >= 0
    z_cible : une hauteur (saut unique de 0 à z_cible à t = 0, cas de l'entraînement) ou une liste de paliers
    ((t_0, h_0), (t_1, h_1), ...) comme MISSION_FIG1. Vitesse de consigne nulle au début de chaque saut.
    Formule exacte en t, sans mémoire : à recopier telle quelle dans mission() de rl_controller.m."""
    t = _t_pas(t)
    p = _paliers(z_cible)
    z = p[0][1]
    for (_, h0), (ti, hi) in zip(p, p[1:]):
        if t >= ti:
            s = t - ti
            z += (hi - h0) * (1.0 - (1.0 + s / TAU_REF) * np.exp(-s / TAU_REF))
    return float(z)


def vitesse_mission(t, z_cible):
    """Vitesse de la consigne lissée (dérivée exacte de mission) : somme_i (h_i - h_(i-1)) * s_i / TAU_REF^2 * exp(-s_i/TAU_REF).
    Nulle au début de chaque saut, maximale à s_i = TAU_REF, puis tend vers 0. À recopier dans rl_controller.m."""
    t = _t_pas(t)
    p = _paliers(z_cible)
    v = 0.0
    for (_, h0), (ti, hi) in zip(p, p[1:]):
        if t >= ti:
            s = t - ti
            v += (hi - h0) * s / TAU_REF ** 2 * np.exp(-s / TAU_REF)
    return float(v)


def recompense(e, e_v, du):
    """Récompense d'un pas. e : écart de position à la consigne lissée [m] ; e_v : écart de vitesse [m/s] ;
    du : variation de commande. Chaque terme est borné : un vol raté coûte au pire ~ -100, un vol parfait rapporte ~ +200."""
    return (W_FIN * np.exp(-(e / SIG_FIN) ** 2)
            + W_LARGE * np.exp(-(e / SIG_LARGE) ** 2)
            - W_LOIN * min(abs(e), E_LOIN) / E_LOIN
            - W_VIT * (1.0 - np.exp(-(e_v / SIG_VIT) ** 2))
            - LAMBDA_DU * du ** 2)


def observation(m, int_e, v_ref):
    """Observation de l'agent : fonction des SEULES mesures m, de l'intégrale et de la vitesse de consigne v_ref
    (formule exacte en t). Même calcul, dans le même ordre, que observation() de rl_controller.m."""
    return np.array([
        (m["z_ref"] - m["z"]) / 0.5,   # erreur de suivi
        m["vz"] / 1.0,                 # vitesse verticale
        m["az_prev"] / 10.0,           # accélération du pas précédent
        m["u_prev"],                   # commande du pas précédent
        int_e / I_MAX,                 # intégrale (conditionnelle) de l'erreur, dans [-1, 1]
        (v_ref - m["vz"]) / 1.0,       # erreur de vitesse par rapport à la consigne lissée
    ], dtype=np.float32)


def maj_integrale(int_e, e):
    """Une mise à jour de l'intégrale : seulement si |e| < E_INT (régime établi), puis bornée à +-I_MAX."""
    if abs(e) < E_INT:
        int_e += e * TS
    return float(np.clip(int_e, -I_MAX, I_MAX))


class DroneEnv(gym.Env):
    def __init__(self, z_ref_fixe=None, sigma_z=0.0, sigma_vz=0.0, sigma_az=0.0, model_path=MODEL_PATH):
        super().__init__()
        self.model = DroneGRU.load(model_path)
        # écarts-types du bruit de MESURE (0 : sans bruit, obligatoire pour export_for_matlab)
        self.sigma = {'sigma_z': sigma_z, 'sigma_vz': sigma_vz, 'sigma_az': sigma_az}
        self.z_ref_fixe = z_ref_fixe          # None : hauteur cible aléatoire ; un nombre : hauteur imposée ; une liste de paliers : mission (export)
        # self.z_ref (fixé au reset) est la hauteur CIBLE ; la consigne vue par le contrôleur est mission(t, self.z_ref)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(6,), dtype=np.float32)

    def _mesures(self):
        """Ce que Simulink donne au contrôleur au pas k. Le bruit ne touche que les mesures, jamais l'état physique."""
        t = self.k * TS
        sigma = self.sigma
        b = lambda s: float(self.np_random.normal(0.0, s)) if s > 0 else 0.0   # bruit gaussien (0 si s = 0)
        return {"t": t,
                "z": self.z + b(sigma['sigma_z']),
                "vz": self.vz + b(sigma['sigma_vz']),
                # au pas 0, az_prev est la valeur initiale du UnitDelay de Simulink : exactement 0, pas une mesure
                "az_prev": 0.0 if self.k == 0 else self.az_prev + b(sigma['sigma_az']),
                "u_prev": self.u_prev,
                "z_ref": mission(t, self.z_ref)}

    def _obs_info(self):
        """Une mesure, UNE mise à jour de l'intégrale, l'observation. Appelée une fois par pas."""
        m = self._mesures()
        self.int_e = maj_integrale(self.int_e, m["z_ref"] - m["z"])
        return observation(m, self.int_e, vitesse_mission(m["t"], self.z_ref)), {"m": m}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)              # initialise self.np_random : indispensable pour reset(seed=0)
        self.k = 0
        self.z, self.vz = 0.0, 0.0            # état de départ, comme dans Simulink
        self.az = self.model.reset()          # az(0) : drone au repos, ~ -9.81
        self.az_prev = 0.0                    # ce que Simulink mesure au pas 0 (UnitDelay initialisé à 0)
        self.u_prev = 0.0                     # idem
        if self.z_ref_fixe is None:
            self.z_ref = float(self.np_random.uniform(*Z_REF_RANGE))   # tirage uniforme avec self.np_random (pas np.random !)
        elif np.isscalar(self.z_ref_fixe):
            self.z_ref = float(self.z_ref_fixe)
        else:
            self.z_ref = _paliers(self.z_ref_fixe)   # mission à plusieurs paliers, ex. MISSION_FIG1
        self.int_e = 0.0                      # comme le persistent int_e = 0 de rl_controller.m
        return self._obs_info()

    def step(self, action):
        u = float(np.clip(np.asarray(action).reshape(-1)[0], -1.0, 1.0))
        du = u - self.u_prev                                      # variation de commande, AVANT la mise à jour de u_prev

        az_next = self.model.step(u)                              # le GRU : u(k) -> az(k+1)
        vz_next = self.vz + TS / 2 * (self.az + az_next)          # trapèzes sur az
        z_next = self.z + TS / 2 * (self.vz + vz_next)            # trapèzes sur vz

        # Mise à jour de l'état pour le pas suivant
        self.az_prev = self.az                # az(k), ce que Simulink mesurera au pas k+1
        self.u_prev = u
        self.z, self.vz, self.az = z_next, vz_next, az_next
        self.k += 1

        # Récompense : état VRAI après le pas (jamais une mesure bruitée) comparé à la consigne lissée au même instant
        t = self.k * TS
        e = mission(t, self.z_ref) - self.z
        e_v = vitesse_mission(t, self.z_ref) - self.vz
        reward = recompense(e, e_v, du)

        terminated = False                    # pas d'arrêt anticipé (voir l'explication de l'étape 3)
        truncated = self.k >= N_STEPS         # 10 s écoulées
        obs, info = self._obs_info()
        return obs, float(reward), terminated, truncated, info
