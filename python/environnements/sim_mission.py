"""
MISSION FINALE : décollage, scan 360° en 10 s, poursuite d'une voiture (trajectoire courbe, vitesse variable) et
atterrissage lent sur la voiture. Tous les contrôleurs sont des agents PPO déjà entraînés (z, lacet, x sur 2 axes).

    python python/environnements/sim_mission.py          ->  resultats/mission/ (vol nominal, robustesse, figures, json)

Contenu :
  - Superviseur : machine à états à 6 phases, décisions prises sur les MESURES (« une fois stable ») ;
  - ControleurMission : superviseur + agents + LQ d'attitude + anticipation 1/cos + mélangeur. Miroir EXACT de
    rl_controller_mission.m (EI_RL/simulink_RL/mission/) : même entrée v = [t ; 18 sorties de UAV_model], même sortie y ;
  - Drone3D : simulateur assemblé à partir des modèles identifiés (GRU de z, GRU du lacet, ARX theta / phi + LQ,
    translation dans le repère du sol tournée par le psi RÉEL), qui produit les 18 mesures (bruit optionnel).

Phases :
  0 STABILISATION  z_ref = 0, cap 0, position (0, 0)                     -> 2 s écoulées
  1 MONTÉE         z_ref : saut lissé 0 -> 0,5 m                          -> |z - 0,5| < 2 cm et |vz| < 5 cm/s pendant 1 s
  2 SCAN           psi_ref : un tour à 2 pi / 10 rad/s (départ et arrêt lissés)
                                                                          -> 10 s écoulées, |psi - 2 pi| < 0,05, |wz| < 0,1
  3 POURSUITE      p_ref = p_voiture + D0 (1 + s/TAU) e^(-s/TAU), v_ref = v_voiture - D0 s/TAU² e^(-s/TAU)
                   (D0 = écart drone - voiture à la fin du scan ; TAU = 1 s, le lissage vu à l'entraînement)
                                                                          -> |p - p_voiture| < 5 cm et |v - v_voiture|
                                                                             < 10 cm/s pendant 2 s d'affilée
  4 DESCENTE       z_ref descend à 0,1 m/s (vz_ref = -0,1) ; PAUSE si |p - p_voiture| > 10 cm   -> z - H_V < 2 cm
  5 POSÉ           z_ref = H_V, suivi de la voiture maintenu (UAV_model n'a pas de sol)
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # dossier python/ (chemins.py)
from chemins import MATLAB, PYTHON, RESULTATS, RL_INTERFACE  # noqa: E402
from env_z import MODEL_PATH as GRU_Z  # noqa: E402
from identification_z import DroneGRU  # noqa: E402
from identification_psi import LacetGRU  # noqa: E402
from env_x import charger_boucle_interne  # noqa: E402
from env_psi import MODEL_PATH as GRU_PSI  # noqa: E402
import voiture as V  # noqa: E402

sys.path.insert(0, str(RL_INTERFACE))
from export_policy import _policy_matrices  # noqa: E402  (fichier fourni, non modifié)

TS = 0.05
SORTIE = RESULTATS / "mission"
AGENTS_RETENUS = dict(z="resultats/agent_z_ppo", psi="resultats/agent_psi_ppo",
                      x="resultats/agent_x_ppo_saturation")
SIGMA_G01 = 0.0045                 # écart-type du bruit de chaque sortie de UAV_model à g_noise = 0,1

# --- paramètres de la mission (À RECOPIER dans rl_controller_mission.m)
T_STAB = 2.0                       # durée de la stabilisation au sol [s]
Z_VOL = 0.5                        # altitude de vol [m]
TAU_Z = 0.35                       # lissage du saut de montée (celui de l'entraînement de l'agent z) [s]
TOL_Z, TOL_VZ, N_STAB_Z = 0.02, 0.05, 20        # « stable » : |z - Z_VOL| < 2 cm, |vz| < 5 cm/s pendant 20 pas (1 s)
T_SCAN = 10.0                      # durée du tour [s]
W_SCAN = 2 * np.pi / T_SCAN        # 0,628 rad/s
TAU_W = 0.35                       # lissage du départ et de l'arrêt du scan (celui de l'entraînement du lacet) [s]
TOL_PSI, TOL_WZ = 0.05, 0.1        # fin du scan : |psi - 2 pi| < 0,05 rad et |wz| < 0,1 rad/s
TAU_APP = 1.0                      # approche : 2e ordre critique, comme le lissage des sauts vus par l'agent x [s]
# --- cap pendant la poursuite (option cap = "voiture") : psi_ref tourne vers la voiture tant qu'elle est loin, puis
#     prend le cap de la voiture quand le drone est au-dessus ; le posé attend en plus |psi - cap voiture| < TOL_CAP
D_LOIN, D_PROCHE = 2.0, 1.0       # au-delà de D_LOIN : viser la voiture ; en deçà de D_PROCHE : cap de la voiture [m]
K_CAP, W_CAP_MAX = 1.0, 0.5       # psi_ref rejoint le cap visé en ~1 s, à 0,5 rad/s au plus
TOL_CAP = 0.05                    # alignement de cap avant la descente [rad] (3°)
TOL_P, TOL_V, N_ALIGNE = 0.05, 0.10, 40         # alignement : 5 cm, 10 cm/s, pendant 40 pas (2 s)
V_DESC = 0.1                       # vitesse de descente [m/s]
TOL_P_DESC = 0.10                  # descente en pause si l'écart horizontal dépasse 10 cm
H_V = 0.0                          # hauteur de la plateforme de la voiture [m]
TOL_POSE = 0.02                    # contact : z - H_V < 2 cm
# --- contrôleur (identiques aux étapes précédentes)
THETA_MAX, U_PSI_MAX, U_EQ, U_MAX_ATT = 0.2, 0.03, 0.39, 0.5
I_MAX, E_INT = 0.5, 0.1
KP_PD, KD_PD = 0.3, 0.45
# Configuration FINALE (mission « cap ») : approche tau = 1 s, cap vers la voiture puis dans son sens, biais de l'agent
# x corrigé. CONFIG_V1 = première version (cap fixe à 2 pi, biais brut), gardée pour comparaison.
CONFIG_FINALE = dict(tau_app=1.0, cap="voiture", corr_biais=True)
CONFIG_V1 = dict(tau_app=1.0, cap="fixe", corr_biais=False)
NOMS_PHASES = ("stabilisation", "montée", "scan", "poursuite", "descente", "posé")


def charger_politiques(chemins=AGENTS_RETENUS):
    """Matrices des réseaux (format policy.mat) : le calcul numpy ci-dessous est celui de rl_controller.m."""
    from stable_baselines3 import PPO
    return {k: _policy_matrices(PPO.load(PYTHON / c / "best_model", device="cpu")) for k, c in chemins.items()}


def reseau(P, obs):
    x = np.asarray(obs, np.float64).reshape(-1, 1)
    for i, a in enumerate(P["activations"].split(","), start=1):
        x = P[f"W{i}"] @ x + P[f"b{i}"]
        x = np.maximum(x, 0) if a == "relu" else (np.tanh(x) if a == "tanh" else x)
    return float(np.clip(x[0, 0], P["action_low"][0], P["action_high"][0]))


def wrap(a):
    """Angle ramené dans [-pi, pi[ (même formule que mod(a + pi, 2 pi) - pi en MATLAB)."""
    return (a + np.pi) % (2 * np.pi) - np.pi


def lisse(s, tau):
    """Réponse indicielle d'un 2e ordre critique et sa dérivée (s >= 0)."""
    e = np.exp(-s / tau)
    return 1.0 - (1.0 + s / tau) * e, s / tau ** 2 * e


def integre(i, e):
    if abs(e) < E_INT:
        i += e * TS
    return float(np.clip(i, -I_MAX, I_MAX))


class ControleurMission:
    """v = [t ; x y z vx vy vz phi theta psi wx wy wz ax ay az wdx wdy wdz]  ->  y (16 sorties) :
    [u1..u4 ; phase ; z_ref ; psi_ref ; x_ref ; y_ref ; x_voiture ; y_voiture ; theta_ref ; phi_ref ; u_z ; u_theta ; u_phi]"""

    def __init__(self, P, table_voiture, mode="rl", tau_app=TAU_APP, cap="fixe", corr_biais=False):
        """mode : 'rl' (agent x) ou 'pd' ; tau_app : constante de l'approche [s] ; cap : 'fixe' (psi = 2 pi après le scan)
        ou 'voiture' (vers la voiture, puis dans son sens avant de se poser) ; corr_biais : retranche à l'agent x sa
        sortie à l'équilibre a(0) = reseau(obs = 0) (~0,035 : biais que l'intégrale mettait plusieurs secondes à effacer)."""
        self.P, self.voit, self.mode, self.tau_app, self.cap = P, table_voiture, mode, tau_app, cap
        self.psi_mem = 0.0
        self.a0_x = reseau(P["x"], np.zeros(7)) if corr_biais else 0.0
        K = charger_boucle_interne()["K"]
        self.K = K
        self.phase, self.n_stab, self.n_al = 0, 0, 0
        self.k1 = self.k2 = self.k3 = self.k4 = -1
        self.D0 = np.zeros(2)
        self.z_desc = Z_VOL
        self.int_z = self.int_psi = self.int_av = self.int_co = 0.0
        self.a_av = self.a_co = 0.0
        self.az_prev = self.wdz_prev = self.u_z_prev = self.a_psi_prev = self.wdy_prev = self.wdx_prev = 0.0
        self.Uth, self.Uph = np.zeros(5), np.zeros(5)

    # ------------------------------------------------------------------ superviseur
    def superviseur(self, k, m):
        x, y, z, vx, vy, vz, psi, wz = m[0], m[1], m[2], m[3], m[4], m[5], m[8], m[11]
        xv, yv, vxv, vyv = V.lire(self.voit, k * TS)
        cap_v, w_v = V.lire_cap(self.voit, k * TS)
        # transitions (sur les mesures de CE pas)
        if self.phase == 0 and k * TS >= T_STAB:
            self.phase, self.k1 = 1, k
        elif self.phase == 1:
            self.n_stab = self.n_stab + 1 if (abs(z - Z_VOL) < TOL_Z and abs(vz) < TOL_VZ) else 0
            if self.n_stab >= N_STAB_Z:
                self.phase, self.k2 = 2, k
        elif self.phase == 2:
            if (k - self.k2) * TS >= T_SCAN and abs(psi - 2 * np.pi) < TOL_PSI and abs(wz) < TOL_WZ:
                self.phase, self.k3 = 3, k
                self.D0 = np.array([x - xv, y - yv])
        elif self.phase == 3:
            ok = np.hypot(x - xv, y - yv) < TOL_P and np.hypot(vx - vxv, vy - vyv) < TOL_V
            if self.cap == "voiture":
                ok = ok and abs(wrap(psi - cap_v)) < TOL_CAP
            self.n_al = self.n_al + 1 if ok else 0
            if self.n_al >= N_ALIGNE:
                self.phase, self.k4 = 4, k
                self.z_desc = Z_VOL
        elif self.phase == 4 and z - H_V < TOL_POSE:
            self.phase = 5
        # consignes
        if self.phase == 0:
            z_ref, vz_ref = 0.0, 0.0
        elif self.phase <= 3:
            r, dr = lisse((k - self.k1) * TS, TAU_Z)
            z_ref, vz_ref = Z_VOL * r, Z_VOL * dr
        elif self.phase == 4:
            descend = np.hypot(x - xv, y - yv) < TOL_P_DESC and self.z_desc > H_V
            vz_ref = -V_DESC if descend else 0.0
            z_ref = self.z_desc
            if descend:
                self.z_desc = max(self.z_desc - V_DESC * TS, H_V)
        else:
            z_ref, vz_ref = H_V, 0.0
        if self.phase < 2:
            psi_ref, w_ref = 0.0, 0.0
        else:
            s = (k - self.k2) * TS
            r1, _ = lisse(s, TAU_W)
            S1 = s - 2 * TAU_W + (s + 2 * TAU_W) * np.exp(-s / TAU_W)        # intégrale du saut lissé
            psi_ref, w_ref = W_SCAN * S1, W_SCAN * r1
            if s >= T_SCAN:
                s2 = s - T_SCAN
                r2, _ = lisse(s2, TAU_W)
                psi_ref -= W_SCAN * (s2 - 2 * TAU_W + (s2 + 2 * TAU_W) * np.exp(-s2 / TAU_W))
                w_ref -= W_SCAN * r2
            if self.cap == "voiture" and self.phase >= 3:
                if k == self.k3:
                    self.psi_mem = psi_ref                      # départ : fin du scan (2 pi)
                d = np.hypot(xv - x, yv - y)
                beta = float(np.clip((d - D_PROCHE) / (D_LOIN - D_PROCHE), 0.0, 1.0))
                vise = cap_v + beta * wrap(np.arctan2(yv - y, xv - x) - cap_v)
                w_ref = float(np.clip(K_CAP * wrap(vise - self.psi_mem) + (1 - beta) * w_v, -W_CAP_MAX, W_CAP_MAX))
                psi_ref = self.psi_mem
                self.psi_mem += w_ref * TS
        if self.phase < 3:
            p_ref, v_ref = np.zeros(2), np.zeros(2)
        else:
            s = (k - self.k3) * TS
            ta = self.tau_app
            e = np.exp(-s / ta)
            p_ref = np.array([xv, yv]) + self.D0 * (1 + s / ta) * e
            v_ref = np.array([vxv, vyv]) - self.D0 * s / ta ** 2 * e
        return z_ref, vz_ref, psi_ref, w_ref, p_ref, v_ref, (xv, yv)

    # ------------------------------------------------------------------ un pas
    def __call__(self, v):
        t, m = v[0], np.asarray(v[1:], float)
        k = int(round(t / TS))
        x, y, z, vx, vy, vz, phi, theta, psi = m[:9]
        wx, wy, wz, az, wdx, wdy, wdz = m[9], m[10], m[11], m[14], m[15], m[16], m[17]
        z_ref, vz_ref, psi_ref, w_ref, p_ref, v_ref, pv = self.superviseur(k, m)
        P = self.P
        # z (env_z.observation)
        e = z_ref - z
        self.int_z = integre(self.int_z, e)
        u_z = reseau(P["z"], [e / 0.5, vz, self.az_prev / 10, self.u_z_prev, self.int_z / I_MAX, vz_ref - vz])
        # lacet (env_psi.observation)
        e = psi_ref - psi
        self.int_psi = integre(self.int_psi, e)
        a_psi = reseau(P["psi"], [e / 0.5, wz, self.wdz_prev / 5, self.a_psi_prev, self.int_psi / I_MAX, w_ref - wz])
        # position (sim_xy.ControleurXY) : rotation de -psi puis agent x sur les deux axes du drone
        c, s = np.cos(psi), np.sin(psi)
        ex, ey = p_ref[0] - x, p_ref[1] - y
        e_av, e_co = c * ex + s * ey, -s * ex + c * ey
        v_av, v_co = c * vx + s * vy, -s * vx + c * vy
        r_av, r_co = c * v_ref[0] + s * v_ref[1], -s * v_ref[0] + c * v_ref[1]
        self.int_av, self.int_co = integre(self.int_av, e_av), integre(self.int_co, e_co)
        o_av = [e_av / 2, v_av, theta / THETA_MAX, wy, self.a_av, self.int_av / I_MAX, r_av - v_av]
        o_co = [e_co / 2, v_co, -phi / THETA_MAX, -wx, self.a_co, self.int_co / I_MAX, r_co - v_co]
        if self.mode == "pd":
            pd = lambda o: float(np.clip(KP_PD * 2 * o[0] + KD_PD * o[6], -THETA_MAX, THETA_MAX)) / THETA_MAX
            self.a_av, self.a_co = pd(o_av), pd(o_co)
        else:
            self.a_av = float(np.clip(reseau(P["x"], o_av) - self.a0_x, -1, 1))
            self.a_co = float(np.clip(reseau(P["x"], o_co) - self.a0_x, -1, 1))
        theta_ref, phi_ref = THETA_MAX * self.a_av, -THETA_MAX * self.a_co
        # LQ d'attitude, anticipation, mélangeur (rl_controller_xy.m)
        u_th = float(np.clip(-self.K @ np.r_[theta - theta_ref, wy, wdy, self.wdy_prev, self.Uth], -U_MAX_ATT, U_MAX_ATT))
        u_ph = float(np.clip(-self.K @ np.r_[phi - phi_ref, wx, wdx, self.wdx_prev, self.Uph], -U_MAX_ATT, U_MAX_ATT))
        u_psi = U_PSI_MAX * a_psi
        u_zc = u_z + U_EQ * (1 / (np.cos(theta) * np.cos(phi)) - 1)
        u = np.clip(u_zc + np.array([u_psi - u_th, -u_psi + u_ph, u_psi + u_th, -u_psi - u_ph]), -1, 1)
        self.Uth = np.r_[(u[2] - u[0]) / 2, self.Uth[:4]]
        self.Uph = np.r_[(u[1] - u[3]) / 2, self.Uph[:4]]
        self.az_prev, self.wdz_prev, self.u_z_prev, self.a_psi_prev = az, wdz, u_z, a_psi
        self.wdy_prev, self.wdx_prev = wdy, wdx
        return np.r_[u, self.phase, z_ref, psi_ref, p_ref, pv, theta_ref, phi_ref, u_z, u_th, u_ph]


class Drone3D:
    """Simulateur assemblé : GRU z (u_z -> az), GRU lacet (u_psi -> wdz), ARX theta / phi (u diff. -> wd), translation
    a_sol = R(psi) [G_X theta ; G_Y phi]. Les entrées sont les 4 commandes moteur (on en extrait u_z, u_psi, u_th, u_ph
    comme le mélangeur inverse) ; u_z est la commande AVANT anticipation 1/cos (le GRU a été identifié à plat)."""

    def __init__(self, sigma=0.0, graine=0):
        p = charger_boucle_interne()
        self.p = p
        with open(PYTHON / "modeles" / "attitude" / "results.json", encoding="utf-8") as f:
            ty = json.load(f)["translation_y"]
        self.gx, self.fx, self.gy, self.fy = p["g_x"], p["frot_x"], float(ty["gain_angle"]), float(ty["frottement"])
        self.gz, self.gpsi = DroneGRU.load(GRU_Z), LacetGRU.load(GRU_PSI)
        self.z = self.vz = 0.0
        self.az = self.gz.reset()
        self.psi = self.wz = 0.0
        self.wdz = self.gpsi.reset()
        self.att = {c: dict(ang=0.0, w=0.0, wd=0.0, wd_prev=0.0, U=np.zeros(5)) for c in ("th", "ph")}
        self.pos, self.vit, self.acc = np.zeros(2), np.zeros(2), np.zeros(2)
        self.rng, self.sigma = np.random.default_rng(graine), sigma

    def _att(self, c, u):
        a, p = self.att[c], self.p
        wd = p["a"][0] * a["wd"] + p["a"][1] * a["wd_prev"] + p["b"][0] * u + p["b"][1:] @ a["U"]
        w = a["w"] + TS / 2 * (a["wd"] + wd)
        a["ang"] += TS / 2 * (a["w"] + w)
        a["U"] = np.r_[u, a["U"][:4]]
        a["wd_prev"], a["wd"], a["w"] = a["wd"], wd, w

    def mesures(self, t):
        th, ph = self.att["th"], self.att["ph"]
        vrai = np.array([self.pos[0], self.pos[1], self.z, self.vit[0], self.vit[1], self.vz, ph["ang"], th["ang"],
                         self.psi, ph["w"], th["w"], self.wz, self.acc[0], self.acc[1], self.az, ph["wd"], th["wd"],
                         self.wdz])
        b = self.rng.normal(0.0, self.sigma, 18) if self.sigma > 0 else 0.0
        return np.r_[t, vrai + b], vrai

    def avance(self, u, u_z):
        u_th, u_ph = (u[2] - u[0]) / 2, (u[1] - u[3]) / 2
        u_psi = (u[0] + u[2] - u[1] - u[3]) / 4
        az = self.gz.step(u_z)
        vz = self.vz + TS / 2 * (self.az + az)
        self.z += TS / 2 * (self.vz + vz)
        self.vz, self.az = vz, az
        wdz = self.gpsi.step(u_psi)
        wz = self.wz + TS / 2 * (self.wdz + wdz)
        self.psi += TS / 2 * (self.wz + wz)
        self.wz, self.wdz = wz, wdz
        self._att("th", u_th); self._att("ph", u_ph)
        c, s = np.cos(self.psi), np.sin(self.psi)
        a_b = np.array([self.gx * self.att["th"]["ang"], self.gy * self.att["ph"]["ang"]])
        a = np.array([c * a_b[0] - s * a_b[1], s * a_b[0] + c * a_b[1]]) + np.array([self.fx, self.fy]) * self.vit
        v = self.vit + TS / 2 * (self.acc + a)
        self.pos = self.pos + TS / 2 * (self.vit + v)
        self.vit, self.acc = v, a


def vol(P, table=None, mode="rl", sigma=0.0, graine=0, t_fin=60.0, tau_app=TAU_APP, cap="fixe", corr_biais=False):
    table = V.trajectoire(t_fin=t_fin + 1) if table is None else table
    ctl, dr = ControleurMission(P, table, mode, tau_app, cap, corr_biais), Drone3D(sigma, graine)
    L = {k: [] for k in ("v", "y", "vrai")}
    for k in range(int(round(t_fin / TS))):
        v, vrai = dr.mesures(k * TS)
        y = ctl(v)
        L["v"].append(v); L["y"].append(y); L["vrai"].append(vrai)
        dr.avance(y[:4], y[13])
    return {k: np.array(a) for k, a in L.items()}


def bilan(L):
    """Instants des phases, écarts au contact et pendant la descente (état VRAI)."""
    t, ph, vr, Y = L["v"][:, 0], L["y"][:, 4], L["vrai"], L["y"]
    d = {f"t_{NOMS_PHASES[i]}": (float(t[np.argmax(ph >= i)]) if (ph >= i).any() else None) for i in range(1, 6)}
    ex, ey = vr[:, 0] - Y[:, 9], vr[:, 1] - Y[:, 10]
    eh = np.hypot(ex, ey)
    if d["t_posé"] is not None:
        i = int(np.argmax(ph >= 5))
        d.update(ecart_contact_m=float(eh[i]), vz_contact_ms=float(vr[i, 5]))
    desc = ph == 4
    d.update(ecart_max_descente_m=float(eh[desc].max()) if desc.any() else None,
             pauses_descente_s=float(np.sum(desc & (np.abs(np.diff(np.r_[Y[:, 5], Y[-1, 5]])) < 1e-9)) * TS)
             if desc.any() else None,
             derive_scan_m=float(np.hypot(vr[ph == 2, 0], vr[ph == 2, 1]).max()),
             erreur_z_vol_m=float(np.abs(vr[(ph >= 2) & (ph <= 3), 2] - Z_VOL).max()),
             theta_max=float(np.abs(vr[:, 7]).max()), phi_max=float(np.abs(vr[:, 6]).max()))
    return d


def bilan_complet(L, table):
    """bilan() + vitesse horizontale relative drone - voiture au contact."""
    d = bilan(L)
    if d["t_posé"] is not None:
        i = int(np.argmax(L["y"][:, 4] >= 5))
        _, _, vxv, vyv = V.lire(table, L["v"][i, 0])
        d["vitesse_relative_horiz_ms"] = float(np.hypot(L["vrai"][i, 3] - vxv, L["vrai"][i, 4] - vyv))
    return d


def figure(vols, table, fichier, titre):
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(2, 2, 1)
    ax.plot(table["x"][:1201], table["y"][:1201], "k", lw=3, alpha=0.25, label="voiture")
    for nom, L in vols.items():
        ph = L["y"][:, 4]
        ax.plot(L["vrai"][:, 0], L["vrai"][:, 1], lw=1.2, label=nom)
        for i, mk in ((3, "o"), (4, "v"), (5, "*")):
            if (ph >= i).any():
                j = int(np.argmax(ph >= i)); ax.plot(L["vrai"][j, 0], L["vrai"][j, 1], mk, ms=8, color="k")
    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]"); ax.axis("equal"); ax.grid(True)
    ax.set_title("Vue de dessus (o : début poursuite, v : début descente, * : posé)"); ax.legend(fontsize=8)
    ax = fig.add_subplot(2, 2, 2)
    for nom, L in vols.items():
        t = L["v"][:, 0]
        ax.plot(t, L["vrai"][:, 2], lw=1.2, label=f"z, {nom}")
    L0 = next(iter(vols.values())); t = L0["v"][:, 0]
    ax.plot(t, L0["y"][:, 5], "k:", label="z_ref")
    ax2 = ax.twinx(); ax2.step(t, L0["y"][:, 4], "C7", where="post", lw=1); ax2.set_ylabel("phase")
    ax2.set_yticks(range(6)); ax2.set_yticklabels(NOMS_PHASES, fontsize=7)
    ax.set_xlabel("t [s]"); ax.set_ylabel("z [m]"); ax.grid(True); ax.legend(fontsize=8, loc="center left")
    ax.set_title("Altitude et phases de la mission")
    ax = fig.add_subplot(2, 2, 3)
    for nom, L in vols.items():
        t = L["v"][:, 0]
        e = np.hypot(L["vrai"][:, 0] - L["y"][:, 9], L["vrai"][:, 1] - L["y"][:, 10])
        ax.semilogy(t, np.maximum(e, 1e-4), lw=1, label=nom)
    ax.axhline(TOL_P, color="k", ls=":", lw=1); ax.axhline(TOL_P_DESC, color="k", ls="--", lw=1)
    ax.set_ylim(1e-3, 20); ax.set_xlabel("t [s]"); ax.set_ylabel("|drone - voiture| horizontal [m]"); ax.grid(True, which="both")
    ax.set_title("Écart horizontal à la voiture (pointillés : 5 cm alignement, tirets : 10 cm pause)"); ax.legend(fontsize=8)
    ax = fig.add_subplot(2, 2, 4)
    t = L0["v"][:, 0]
    ax.plot(t, L0["y"][:, 6], "k:", label="ψ_ref"); ax.plot(t, L0["vrai"][:, 8], label="ψ")
    ax.plot(table["t"][:len(t)], np.unwrap(table["cap"][:len(t)]) + 2 * np.pi, color="0.6", lw=2, alpha=0.6,
            label="cap de la voiture (+2π)")
    ax.set_ylabel("ψ [rad]"); ax.grid(True)
    ax3 = ax.twinx(); ax3.plot(t, L0["vrai"][:, 7], "C1", lw=0.9, label="θ"); ax3.plot(t, L0["vrai"][:, 6], "C2", lw=0.9, label="φ")
    ax3.set_ylabel("θ, φ [rad]"); ax.set_xlabel("t [s]")
    ax.legend(loc="upper left", fontsize=8); ax3.legend(loc="lower right", fontsize=8)
    ax.set_title(f"Cap et inclinaisons ({next(iter(vols))})")
    fig.suptitle(titre); fig.tight_layout(); fig.savefig(fichier, dpi=115); plt.close(fig)


def main(cfg=CONFIG_FINALE, suffixe="_cap"):
    """suffixe '' + CONFIG_V1 : reproduit la première version de la mission."""
    SORTIE.mkdir(parents=True, exist_ok=True)
    P = charger_politiques()
    table = V.trajectoire(t_fin=61.0)
    vols, R = {}, {}
    for mode, nom in (("rl", "agents"), ("pd", "PD x/y")):
        for sig, gn in ((0.0, "g=0"), (SIGMA_G01, "g=0,1")):
            L = vol(P, table, mode, sig, graine=1, **cfg)
            vols[f"{nom}, {gn}"] = L
            R[f"{nom}, {gn}"] = bilan_complet(L, table)
    for k, d in R.items():
        print(k, {a: (round(b, 3) if isinstance(b, float) else b) for a, b in d.items()})
    # robustesse : voitures tirées au hasard, avec bruit
    rob = []
    for g in range(12):
        p = V.voiture_aleatoire(np.random.default_rng(100 + g))
        tab = V.trajectoire(p, t_fin=61.0)
        for mode in ("rl", "pd"):
            d = bilan_complet(vol(P, tab, mode, SIGMA_G01, graine=g, **cfg), tab)
            d.update(graine=100 + g, mode=mode, v_max=float(tab["v"].max()))
            rob.append(d)
    with open(SORTIE / f"resultats_mission{suffixe}.json", "w", encoding="utf-8") as f:
        json.dump(dict(nominal=R, robustesse=rob), f, indent=2, ensure_ascii=False)
    for mode in ("rl", "pd"):
        r = [d for d in rob if d["mode"] == mode]
        pos = [d for d in r if d["t_posé"] is not None]
        print(f"robustesse {mode} : posé {len(pos)}/{len(r)} ; écart au contact max "
              f"{max((d['ecart_contact_m'] for d in pos), default=float('nan')):.3f} m ; vitesse relative max "
              f"{max((d['vitesse_relative_horiz_ms'] for d in pos), default=float('nan')):.3f} m/s ; t posé moyen "
              f"{np.mean([d['t_posé'] for d in pos]):.1f} s")
    figure({k: vols[k] for k in ("agents, g=0", "agents, g=0,1", "PD x/y, g=0,1")}, table, SORTIE / f"sim_mission{suffixe}.png",
           "Mission finale (simulateur assemblé : GRU z, GRU lacet, ARX attitude) : agents PPO z + lacet + x/y")
    np.savez_compressed(SORTIE / f"sim_mission_vols{suffixe}.npz", **{f"{k}_{c}": v for k, L in vols.items() for c, v in L.items()})
    print(f"-> {SORTIE}")


if __name__ == "__main__":
    main()
