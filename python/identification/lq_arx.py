"""
COMMANDE LQ SUR MODÈLE ARX IDENTIFIÉ, pour z et pour le lacet (même méthode que le LQ d'attitude de l'étape C1,
commande_LQ/LQ_ARX_attitude/gains_LQ_ARX_attitude.m). Sert de troisième référence dans les comparaisons
« LQI double intégrateur (double intégrateur) / LQ sur modèle ARX / agent RL » (python/evaluation/comparaisons_LQI_ARX_RL.py).

Principe :
  1. modèle ARX(2,6) de l'accélération à partir de la commande, identifié par moindres carrés :
        acc(k) = a1 acc(k-1) + a2 acc(k-2) + b1 u(k-1) + ... + b6 u(k-6) (+ c pour z : gravité)
     z : sur les données nettoyées (donnees/nettoyees/, les mêmes que le LQI double intégrateur) ;
     lacet : l'ARX déjà identifié à l'étape B1 (python/modeles/psi/results.json, clé « arx ») ;
  2. modèle d'état discret exact (intégrations par trapèzes, comme les environnements) :
        x = [position, vitesse, acc(k), acc(k-1), u(k-1), ..., u(k-5) (, intégrale de l'erreur)] ;
  3. gain par Riccati discrète (dlqr) ; commande u = -K (x - x_consigne), toutes les composantes étant mesurées ou
     mémorisées, sauf acc(k) qui est PRÉDITE par l'ARX à partir du passé (les environnements, comme l'interface RL, ne
     donnent que acc(k-1)) : aucun observateur supplémentaire.
Différence avec le LQI double intégrateur : le modèle contient la dynamique des moteurs (les 6 coefficients b, retard
d'un pas compris) au lieu d'un double intégrateur pur ; les gains restent donc stables à 20 Hz.

    python python/identification/lq_arx.py     # identifie l'ARX de z, affiche les gains et les pôles
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.linalg import solve_discrete_are

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # dossier python/ (chemins.py)
from chemins import DONNEES, MODELES  # noqa: E402

TS = 0.05
FICHIER_ARX_Z = MODELES / "z" / "arx_z.json"


# ------------------------------------------------------------------ identification de l'ARX de z
def identifier_arx_z(n_traj=600, graine=0, na=2, nb=6):
    """ARX avec constante : az(k) = sum a_i az(k-i) + sum b_j u(k-j) + c, sur n_traj trajectoires nettoyées.
    Renvoie a, b, c, u_eq (commande d'équilibre : az = 0) et la RMSE de simulation libre sur 200 autres trajectoires."""
    d = DONNEES / "nettoyees"
    U = np.loadtxt(d / "bdd_in_clean.csv.gz", delimiter=",")
    A = np.loadtxt(d / "bdd_out_clean.csv.gz", delimiter=",")
    idx = np.random.default_rng(graine).permutation(len(U))
    tr, te = idx[:n_traj], idx[n_traj:n_traj + 200]
    lignes, cible = [], []
    for u, y in zip(U[tr], A[tr]):
        for k in range(max(na, nb), len(y)):
            lignes.append(np.r_[y[k - na:k][::-1], u[k - nb:k][::-1], 1.0]); cible.append(y[k])
    th, *_ = np.linalg.lstsq(np.array(lignes), np.array(cible), rcond=None)
    a, b, c = th[:na], th[na:na + nb], float(th[-1])
    u_eq = -c * 1.0 / b.sum() if abs(1 - a.sum()) > 0 else np.nan     # az = 0  <=>  sum(b) u_eq + c = 0
    # simulation libre sur les trajectoires de test (départ au repos, az(0) mesuré)
    err = []
    for u, y in zip(U[te], A[te]):
        yh = np.zeros_like(y); yh[:nb] = y[:nb]
        for k in range(nb, len(y)):
            yh[k] = a @ yh[k - na:k][::-1] + b @ u[k - nb:k][::-1] + c
        err.append(yh[nb:] - y[nb:])
    rmse = float(np.sqrt(np.mean(np.concatenate(err) ** 2)))
    return dict(a=a.tolist(), b=b.tolist(), c=c, u_eq=float(u_eq), rmse_simulation_libre=rmse, n_traj=n_traj)


def charger_arx_z():
    if not FICHIER_ARX_Z.exists():
        FICHIER_ARX_Z.write_text(json.dumps(identifier_arx_z(), indent=2), encoding="utf-8")
    return json.loads(FICHIER_ARX_Z.read_text(encoding="utf-8"))


def charger_arx_psi():
    return json.loads((MODELES / "psi" / "results.json").read_text(encoding="utf-8"))["arx"]


# ------------------------------------------------------------------ modèle d'état et synthèse
def modele_arx(a, b, integrale=False):
    """(A, B) discrets, x = [pos, vit, acc(k), acc(k-1), u(k-1..k-5)] (+ intégrale de pos), entrée u(k).
    acc(k+1) = a1 acc(k) + a2 acc(k-1) + b1 u(k) + b2 u(k-1) + ... + b6 u(k-5) ; vit, pos : trapèzes."""
    n = 10 if integrale else 9
    A, B = np.zeros((n, n)), np.zeros(n)
    Lacc = np.r_[0, 0, a[0], a[1], b[1:6], np.zeros(n - 9)]
    A[2], B[2] = Lacc, b[0]
    A[1] = np.r_[0, 1, TS / 2, 0, np.zeros(n - 4)] + TS / 2 * Lacc; B[1] = TS / 2 * b[0]
    A[0] = np.r_[1, TS / 2, 0, 0, np.zeros(n - 4)] + TS / 2 * A[1]; B[0] = TS / 2 * B[1]
    A[3, 2] = 1.0
    B[4] = 1.0; A[5, 4] = A[6, 5] = A[7, 6] = A[8, 7] = 1.0
    if integrale:
        A[9, 0], A[9, 9] = TS, 1.0
    return A, B.reshape(-1, 1)


def synthese(A, B, q_pos, q_vit, R, q_int=0.0):
    n = A.shape[0]
    Q = np.diag([q_pos, q_vit] + [0.0] * 7 + ([q_int] if n == 10 else []))
    P = solve_discrete_are(A, B, Q, np.array([[R]]))
    return np.linalg.solve(R + B.T @ P @ B, B.T @ P @ A).ravel()


# ------------------------------------------------------------------ contrôleurs (interface act(obs, m) des essais)
class LQIARXz:
    """LQI sur l'ARX de z (avec intégrale : l'équilibre u_eq n'est connu qu'approximativement).
    Mesures utilisées (comme l'agent) : z, vz, az(k-1), commandes passées ; az(k) prédit par l'ARX."""
    Q_POS, Q_VIT, Q_INT, R = 400.0, 20.0, 400.0, 2.0       # réglage : montée ~0,6 s, dépassement < 5 %

    def __init__(self, reglage=None):
        p = charger_arx_z()
        self.a, self.b, self.c, self.u_eq = np.array(p["a"]), np.array(p["b"]), p["c"], p["u_eq"]
        q = reglage or (self.Q_POS, self.Q_VIT, self.R, self.Q_INT)
        self.K = synthese(*modele_arx(self.a, self.b, integrale=True), *q)

    def reset(self, m=None):
        self.U = np.zeros(6)          # u'(k-1..k-6) (écarts à u_eq)
        self.acc = np.zeros(2)        # az(k-1), az(k-2) mesurés
        self.integ = 0.0

    def act(self, obs, m):
        self.acc = np.r_[m["az_prev"], self.acc[0]]
        az_k = self.a @ self.acc + self.b @ self.U + self.c + self.b.sum() * self.u_eq      # prédiction ARX de az(k)
        e = m["z"] - m["z_ref"]
        self.integ = float(np.clip(self.integ + e * TS, -0.5, 0.5))
        x = np.r_[e, m["vz"], az_k, self.acc[0], self.U[:5], self.integ]
        du = float(-self.K @ x)
        u = float(np.clip(self.u_eq + du, -1.0, 1.0))
        self.U = np.r_[u - self.u_eq, self.U[:5]]
        return u


class LQARXpsi:
    """LQ sur l'ARX du lacet (étape B1), sans intégrale (pas de couple parasite), suivi de l'angle ET de la vitesse de
    consigne (x = [psi - psi_ref, wz - w_ref, ...]). Sortie normalisée a = u_psi / U_PSI_MAX, comme l'agent."""
    Q_POS, Q_VIT, R = 100.0, 2.0, 2000.0

    def __init__(self, w_ref_fn=None, u_max=0.03, reglage=None):
        p = charger_arx_psi()
        self.a, self.b = np.array(p["a"]), np.array(p["b"])
        self.K = synthese(*modele_arx(self.a, self.b), *(reglage or (self.Q_POS, self.Q_VIT, self.R)))
        self.u_max, self.w_ref_fn = u_max, w_ref_fn

    def reset(self, m=None):
        self.U = np.zeros(6); self.acc = np.zeros(2)

    def act(self, obs, m):
        self.acc = np.r_[m["wdz_prev"], self.acc[0]]
        wd_k = self.a @ self.acc + self.b @ self.U
        w_ref = self.w_ref_fn(m["t"]) if self.w_ref_fn else 0.0
        x = np.r_[m["psi"] - m["psi_ref"], m["wz"] - w_ref, wd_k, self.acc[0], self.U[:5]]
        u = float(np.clip(-self.K @ x, -self.u_max, self.u_max))
        self.U = np.r_[u, self.U[:5]]
        return u / self.u_max


if __name__ == "__main__":
    p = identifier_arx_z()
    FICHIER_ARX_Z.write_text(json.dumps(p, indent=2), encoding="utf-8")
    print(f"ARX z : a = {np.round(p['a'], 4)}, b = {np.round(p['b'], 3)}, c = {p['c']:.3f}, u_eq = {p['u_eq']:.4f}, "
          f"RMSE simulation libre {p['rmse_simulation_libre']:.3f} m/s²  -> {FICHIER_ARX_Z}")
    for nom, ctl in (("z", LQIARXz()), ("psi", LQARXpsi())):
        A, B = modele_arx(ctl.a, ctl.b, integrale=(nom == "z"))
        print(f"{nom} : K = {np.round(ctl.K, 4)} ; |pôles BF| max {np.abs(np.linalg.eigvals(A - B @ ctl.K[None, :])).max():.3f}")
