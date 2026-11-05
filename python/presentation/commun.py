"""
OUTILS COMMUNS DES ANIMATIONS DE LA PRÉSENTATION (sorties dans rapport/presentation/).

À quoi sert ce module :
  1. ControleurFamille : le contrôleur de la mission (superviseur de sim_mission.ControleurMission, inchangé) avec,
     au choix, l'une des TROIS FAMILLES de lois de commande comparées dans le rapport, sur TOUS les axes :
       "RL"  : agents PPO z, lacet, x (deux axes du drone, biais corrigé) + LQ d'attitude sur l'ARX ;
       "ARX" : LQI sur l'ARX de z, LQ sur l'ARX du lacet (identification/lq_arx.py), PD de position 0,3 / 0,45 avec
               vitesse de consigne + LQ d'attitude sur l'ARX ;
       "LQI" : LQI « double intégrateur » (z, lacet, theta / phi) avec les gains réajustés pour 20 Hz
               (identification/reajuster_gains_LQI.py) et son PD de position 0,23 / 0,28, avec vitesse de consigne
               (version de la mission d'atterrissage LQ).
     Le mélangeur et l'anticipation 1 / cos de la poussée sont les mêmes pour les trois.
  2. mode « suivi » (suivi=True) : PAS d'atterrissage. Le drone part au-dessus de la voiture à 0,5 m et doit y rester
     (p_ref = position de la voiture, v_ref = sa vitesse, cap = cap de la voiture), quelle que soit la conduite.
  3. VOITURE_LIGNE_DROITE, VOITURE_SLALOM : deux conduites « à forts à-coups » (démarrages et freinages brusques,
     marche arrière ; virages serrés enchaînés). Même modèle que voiture.py (unicycle, transitions en S), mais avec des
     transitions très courtes.
  4. vol_famille : un vol sur le simulateur assemblé (sim_mission.Drone3D, modèles identifiés), état VRAI enregistré.
  5. Dessins : drone vu de dessus (4 rotors, flèche de cap), voiture, mise en page claire.
"""
import sys
from pathlib import Path

import numpy as np
from matplotlib import patches, transforms

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # dossier python/ (chemins.py)
from chemins import PRESENTATION  # noqa: E402
from lq_arx import LQARXpsi, LQIARXz  # noqa: E402
from reajuster_gains_LQI import gains_20Hz  # noqa: E402
import sim_mission as M  # noqa: E402
import voiture as V  # noqa: E402

TS = M.TS
PRESENTATION.mkdir(parents=True, exist_ok=True)

FAMILLES = ("RL", "ARX", "LQI")
NOMS = {"RL": "agents RL (PPO)", "ARX": "LQ sur modèle ARX", "LQI": "LQI double intégrateur (réajusté 20 Hz)"}
COUL = {"RL": "tab:blue", "ARX": "tab:green", "LQI": "tab:red"}
KP_CASC, KD_CASC = 0.23, 0.28        # PD de position  (commande_LQ/gains_LQI.m)

# ------------------------------------------------------------------ conduites à forts à-coups
T_CHAUFFE = 2.0     # suivi : 2 s de vol stationnaire NON enregistrées au début (le GRU de z part d'un état de repos :
                    # sans cela, le drone posé à 0,5 m chute de 12 cm pendant 0,5 s) ; la voiture part 2 s après.
# (instant de début de la transition [s depuis le démarrage], durée [s], nouvelle valeur) : transitions de 0,6 à 1 s
# (contre 2 à 4 s pour la voiture de la mission) -> accélérations de pointe 1 à 1,5 m/s².
VOITURE_LIGNE_DROITE = dict(
    t_depart=T_CHAUFFE + 2.0, x0=0.0, y0=0.0, cap0=0.0,
    vitesse=((0.0, 0.8, 0.8),          # démarrage sec : 0 -> 0,8 m/s en 0,8 s
             (4.0, 0.6, 0.0),          # coup de frein : arrêt en 0,6 s
             (6.0, 0.8, -0.5),         # marche arrière
             (8.5, 0.7, 0.0),          # arrêt
             (10.0, 1.0, 1.0),         # nouveau départ, plus vite
             (13.0, 0.8, 0.3),         # ralentissement brusque
             (15.0, 0.6, 0.9),         # réaccélération
             (17.5, 0.7, 0.0)),        # arrêt final
    courbure=(),
)
VOITURE_SLALOM = dict(
    t_depart=T_CHAUFFE + 2.0, x0=0.0, y0=0.0, cap0=0.0,
    vitesse=((0.0, 1.0, 0.6), (11.0, 0.8, 0.45), (17.0, 1.0, 0.7), (21.0, 0.8, 0.0)),
    courbure=((1.5, 0.6, 1 / 1.2), (3.5, 0.6, -1 / 1.2), (5.5, 0.6, 1 / 1.2), (7.5, 0.6, -1 / 1.2),   # slalom serré
              (9.5, 0.6, 0.0),
              (11.5, 0.6, 1 / 0.9), (16.0, 0.6, 0.0),                     # demi-tour serré (rayon 0,9 m)
              (17.5, 0.6, -1 / 1.5), (19.5, 0.6, 0.0)),                  # dernier virage, à droite
)
CONDUITES = {"ligne_droite": ("ligne droite : démarrages, coups de frein, marche arrière", VOITURE_LIGNE_DROITE, 22.0),
             "slalom": ("slalom serré puis demi-tour", VOITURE_SLALOM, 24.0)}   # durée affichée [s]


def table_voiture(p, t_fin):
    """Table de voiture.py (+ accélération longitudinale et latérale, pour les figures)."""
    T = V.trajectoire(p, t_fin=t_fin + 1)
    T["a_long"] = np.r_[0.0, np.diff(T["v"]) / TS]
    T["a_lat"] = T["v"] ** 2 * T["kappa"]
    return T


# ------------------------------------------------------------------ contrôleur : superviseur + famille
class ControleurFamille(M.ControleurMission):
    """Même entrée / sortie que sim_mission.ControleurMission (miroir de rl_controller_mission.m) ; seules les lois de
    commande changent selon la famille. suivi=True : superviseur remplacé par « rester au-dessus de la voiture »."""

    def __init__(self, famille, P, table, suivi=False):
        super().__init__(P, table, mode="pd" if famille == "ARX" else "rl", tau_app=M.TAU_APP, cap="voiture",
                         corr_biais=True)
        self.famille, self.suivi = famille, suivi
        if famille == "ARX":
            self.w_ref = 0.0
            self.lq_z, self.lq_psi = LQIARXz(), LQARXpsi(w_ref_fn=lambda t: self.w_ref)
            self.lq_z.reset(); self.lq_psi.reset()
        elif famille == "LQI":
            self.g_z, self.g_psi, self.g_att = gains_20Hz("z"), gains_20Hz("psi"), gains_20Hz("theta")
            self.I = dict(z=0.0, psi=0.0, th=0.0, ph=0.0)
        if suivi:
            self.phase = 3
            self.psi_mem = 0.0

    def superviseur(self, k, m):
        if not self.suivi:
            return super().superviseur(k, m)
        xv, yv, vxv, vyv = V.lire(self.voit, k * TS)
        cap_v, w_v = V.lire_cap(self.voit, k * TS)
        w_ref = float(np.clip(M.K_CAP * M.wrap(cap_v - self.psi_mem) + w_v, -M.W_CAP_MAX, M.W_CAP_MAX))
        psi_ref = self.psi_mem
        self.psi_mem += w_ref * TS
        return M.Z_VOL, 0.0, psi_ref, w_ref, np.array([xv, yv]), np.array([vxv, vyv]), (xv, yv)

    def __call__(self, v):
        if self.famille == "RL":
            return super().__call__(v)
        t, m = v[0], np.asarray(v[1:], float)
        k = int(round(t / TS))
        x, y, z, vx, vy, vz, phi, theta, psi = m[:9]
        wx, wy, wz, az, wdx, wdy, wdz = m[9], m[10], m[11], m[14], m[15], m[16], m[17]
        z_ref, vz_ref, psi_ref, w_ref, p_ref, v_ref, pv = self.superviseur(k, m)
        c, s = np.cos(psi), np.sin(psi)
        ex, ey = p_ref[0] - x, p_ref[1] - y
        e_av, e_co = c * ex + s * ey, -s * ex + c * ey
        dv_av = c * (v_ref[0] - vx) + s * (v_ref[1] - vy)
        dv_co = -s * (v_ref[0] - vx) + c * (v_ref[1] - vy)
        if self.famille == "ARX":
            u_z = self.lq_z.act(None, dict(az_prev=self.az_prev, z=z, z_ref=z_ref, vz=vz))
            self.w_ref = w_ref
            u_psi = M.U_PSI_MAX * self.lq_psi.act(None, dict(wdz_prev=self.wdz_prev, psi=psi, psi_ref=psi_ref, wz=wz, t=t))
            kp, kd = M.KP_PD, M.KD_PD
        else:                                                    # LQI double intégrateur, gains réajustés
            I = self.I
            e = z_ref - z; I["z"] += e * TS
            kp_, kd_, ki_ = self.g_z
            u_z = float(np.clip(0.38 + kp_ * e - kd_ * vz + ki_ * I["z"], -1.0, 1.0))
            e = psi_ref - psi; I["psi"] += e * TS
            kp_, kd_, ki_ = self.g_psi
            u_psi = float(np.clip(kp_ * e - kd_ * wz + ki_ * I["psi"], -M.U_PSI_MAX, M.U_PSI_MAX))
            kp, kd = KP_CASC, KD_CASC
        theta_ref = float(np.clip(kp * e_av + kd * dv_av, -M.THETA_MAX, M.THETA_MAX))
        phi_ref = -float(np.clip(kp * e_co + kd * dv_co, -M.THETA_MAX, M.THETA_MAX))
        self.a_av, self.a_co = theta_ref / M.THETA_MAX, -phi_ref / M.THETA_MAX
        if self.famille == "ARX":
            u_th = float(np.clip(-self.K @ np.r_[theta - theta_ref, wy, wdy, self.wdy_prev, self.Uth], -M.U_MAX_ATT, M.U_MAX_ATT))
            u_ph = float(np.clip(-self.K @ np.r_[phi - phi_ref, wx, wdx, self.wdx_prev, self.Uph], -M.U_MAX_ATT, M.U_MAX_ATT))
        else:
            kp_, kd_, ki_ = self.g_att
            e = theta_ref - theta; self.I["th"] += e * TS
            u_th = float(np.clip(kp_ * e - kd_ * wy + ki_ * self.I["th"], -M.U_MAX_ATT, M.U_MAX_ATT))
            e = phi_ref - phi; self.I["ph"] += e * TS
            u_ph = float(np.clip(kp_ * e - kd_ * wx + ki_ * self.I["ph"], -M.U_MAX_ATT, M.U_MAX_ATT))
        u_zc = u_z + M.U_EQ * (1 / (np.cos(theta) * np.cos(phi)) - 1)
        u = np.clip(u_zc + np.array([u_psi - u_th, -u_psi + u_ph, u_psi + u_th, -u_psi - u_ph]), -1, 1)
        self.Uth = np.r_[(u[2] - u[0]) / 2, self.Uth[:4]]
        self.Uph = np.r_[(u[1] - u[3]) / 2, self.Uph[:4]]
        self.az_prev, self.wdz_prev, self.u_z_prev = az, wdz, u_z
        self.wdy_prev, self.wdx_prev = wdy, wdx
        return np.r_[u, self.phase, z_ref, psi_ref, p_ref, pv, theta_ref, phi_ref, u_z, u_th, u_ph]


_P = None


def politiques():
    global _P
    if _P is None:
        _P = M.charger_politiques()
    return _P


def vol_famille(famille, table, t_fin, suivi=False, sigma=M.SIGMA_G01, graine=0):
    """Un vol. Renvoie dict t, vrai (18 états VRAIS), y (16 sorties du contrôleur), voiture (x, y, cap, v à chaque pas),
    eh (écart horizontal drone - voiture [m]). suivi=True : t_fin = durée AFFICHÉE ; les T_CHAUFFE premières secondes
    sont simulées puis retirées, et le temps est décalé pour commencer à 0."""
    if suivi:
        t_fin += T_CHAUFFE
    ctl, dr = ControleurFamille(famille, politiques(), table, suivi=suivi), M.Drone3D(sigma, graine)
    if suivi:                                              # départ au-dessus de la voiture, à l'altitude de vol
        dr.pos = np.array([table["x"][0], table["y"][0]]); dr.z = M.Z_VOL
    L = {k: [] for k in ("v", "y", "vrai")}
    for k in range(int(round(t_fin / TS))):
        v, vrai = dr.mesures(k * TS)
        y = ctl(v)
        L["v"].append(v); L["y"].append(y); L["vrai"].append(vrai)
        dr.avance(y[:4], y[13])
        if not np.all(np.isfinite(vrai)) or abs(vrai[2]) > 20:
            break
    R = {k: np.array(a) for k, a in L.items()}
    R["t"] = R["v"][:, 0]
    kv = np.minimum(np.round(R["t"] / TS).astype(int), len(table["t"]) - 1)
    R["voiture"] = {c: table[c][kv] for c in ("x", "y", "cap", "v", "a_long", "a_lat") if c in table}
    R["eh"] = np.hypot(R["vrai"][:, 0] - R["voiture"]["x"], R["vrai"][:, 1] - R["voiture"]["y"])
    if suivi:
        n0 = int(round(T_CHAUFFE / TS))
        R = {k: ({c: a[n0:] for c, a in R[k].items()} if isinstance(R[k], dict) else R[k][n0:]) for k in R}
        R["t"] = R["t"] - T_CHAUFFE
    return R


# ------------------------------------------------------------------ dessins (vue de dessus)
L_VOIT, W_VOIT = 0.45, 0.22        # gabarit de la voiture dessinée [m]
R_DRONE = 0.12                     # demi-envergure du drone dessiné [m]


class DessinVoiture:
    def __init__(self, ax):
        self.ax = ax
        self.caisse = patches.FancyBboxPatch((-L_VOIT / 2, -W_VOIT / 2), L_VOIT, W_VOIT, boxstyle="round,pad=0,rounding_size=0.04",
                                             fc="0.45", ec="k", lw=1, zorder=3)
        self.plateforme = patches.Rectangle((-0.1, -0.08), 0.2, 0.16, fc="w", ec="k", lw=0.8, zorder=4)
        self.phares = patches.Rectangle((L_VOIT / 2 - 0.04, -W_VOIT / 2 + 0.02), 0.04, W_VOIT - 0.04, fc="gold", ec="none", zorder=4)
        for p in (self.caisse, self.plateforme, self.phares):
            ax.add_patch(p)

    def maj(self, x, y, cap):
        tr = transforms.Affine2D().rotate(cap).translate(x, y) + self.ax.transData
        for p in (self.caisse, self.plateforme, self.phares):
            p.set_transform(tr)


class DessinDrone:
    def __init__(self, ax, coul="tab:blue", echelle=1.0, alpha=1.0):
        self.ax, self.e = ax, echelle
        self.bras = [ax.plot([], [], color="k", lw=2, zorder=6, alpha=alpha)[0] for _ in range(2)]
        self.rotors = [patches.Circle((0, 0), 0.045 * echelle, fc=coul, ec="k", zorder=7, alpha=alpha) for _ in range(4)]
        for r in self.rotors:
            ax.add_patch(r)
        self.fleche = ax.annotate("", xy=(0, 0), xytext=(0, 0), zorder=8,
                                  arrowprops=dict(arrowstyle="-|>", lw=1.8, color=coul, alpha=alpha))

    def maj(self, x, y, psi):
        R = R_DRONE * self.e
        for j, ang in enumerate((np.pi / 4, -np.pi / 4)):
            ca, sa = np.cos(psi + ang), np.sin(psi + ang)
            self.bras[j].set_data([x - R * ca, x + R * ca], [y - R * sa, y + R * sa])
        for j, ang in enumerate(np.pi / 4 + np.arange(4) * np.pi / 2):
            self.rotors[j].center = (x + R * np.cos(psi + ang), y + R * np.sin(psi + ang))
        self.fleche.xy = (x + 2.3 * R * np.cos(psi), y + 2.3 * R * np.sin(psi)); self.fleche.set_position((x, y))


def style_clair():
    """Figures en mode CLAIR (fond blanc), quel que soit le style par défaut de la machine."""
    import matplotlib.pyplot as plt
    plt.style.use("default")
    plt.rcParams.update({"figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
                         "font.size": 9})


def enregistrer_gif(an, fichier, fps, dpi=80):
    from matplotlib import animation
    an.save(fichier, writer=animation.PillowWriter(fps=fps), dpi=dpi)
    print(f"-> {fichier} ({Path(fichier).stat().st_size / 1e6:.1f} Mo)")
