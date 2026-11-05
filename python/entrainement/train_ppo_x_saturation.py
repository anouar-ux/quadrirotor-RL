"""
Étape C2.3 bis : réentraînement de l'agent x pour les SATURATIONS (rattraper la voiture avant de se poser dessus).

Le premier agent (train_ppo_x.py) a appris sur des consignes faisables (sauts <= 2 m lissés en 1 s) : il bat le PD sur
le suivi fin, mais pas sur un saut de 5 m où theta_ref sature (il n'a jamais appris à « accélérer à fond puis freiner
au bon moment »). Ici, chaque vol tire au hasard l'une de trois familles de consignes :

  1. PRBS DE LARGE AMPLITUDE (P_PRBS) : la cible bascule entre deux positions écartées de 2 à 8 m, à des instants
     donnés par une séquence binaire pseudo-aléatoire (registre à décalage 6 bits, point de départ aléatoire) de
     période de bit Tb tirée entre 1,5 et 4 s. Paliers courts (le drone n'arrive pas) et longs (il doit freiner).
  2. POURSUITE (P_POURSUITE) : la cible est devant (3 à 9 m) et roule (0,2 à 1 m/s), comme la voiture au moment où
     le drone finit son scan ; parfois elle change ensuite de vitesse. Le drone doit rattraper puis suivre.
  3. MISSIONS NORMALES (le reste) : celles de train_ppo_x.py, pour ne pas perdre la précision du suivi fin.

Récompense : celle de env_x.py (identique pour |e| < 2 m), PROLONGÉE au-delà : sans cela le terme « guidage
lointain » est plat après 2 m d'écart et l'agent ne sait plus s'il se rapproche. On ajoute
    - W_TRES_LOIN * (min(|e|, E_TRES_LOIN) - E_LOIN) / (E_TRES_LOIN - E_LOIN)   pour |e| > E_LOIN = 2 m.

Évaluation : 20 missions normales fixes (précision) + 6 scénarios de saturation fixes (rattrapage) ; le meilleur
modèle est celui qui maximise la SOMME des deux rendements moyens (aucun des deux aspects n'est sacrifié).

    python python/entrainement/train_ppo_x_saturation.py                                   # 4 M transitions depuis zéro (~15-20 min)
    python python/entrainement/train_ppo_x_saturation.py --depart resultats/agent_x_ppo --steps 2000000
                                                                            # reprise de l'agent actuel (plus court)
    python python/entrainement/train_ppo_x_saturation.py --steps 20000 --eval-freq 10000   # essai rapide du script
Sorties : python/resultats/agent_x_ppo_saturation_AAAAMMJJ_HHMM/ (best_model.zip, results.json, learning_curve.png, vols.png).
"""
import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from stable_baselines3 import PPO  # noqa: E402
from stable_baselines3.common.callbacks import BaseCallback  # noqa: E402
from stable_baselines3.common.env_util import make_vec_env  # noqa: E402
from stable_baselines3.common.vec_env import DummyVecEnv  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # dossier python/ (chemins.py)
from chemins import RESULTATS, PYTHON  # noqa: E402  (rend importables les sous-dossiers de python/)
from env_z import TS  # noqa: E402
from reference_pd_x import SIGMA_G01, evaluer, missions_alea, pd_anticipation  # noqa: E402
from env_x import E_LOIN, N_STEPS, THETA_MAX, XEnv, consigne_x, mission_aleatoire, recompense  # noqa: E402

# ------------------------------------------------------------------ réglages
P_PRBS, P_POURSUITE = 0.4, 0.3        # proportions des familles de consignes (le reste : missions normales)
PRBS_ECART = (2.0, 8.0)               # écart entre les deux niveaux de la PRBS [m]
PRBS_TB = (1.5, 4.0)                  # période de bit [s]
POURSUITE_D0 = (3.0, 9.0)             # avance initiale de la cible [m]
POURSUITE_V = (0.2, 1.0)              # vitesse de la cible [m/s]
X_ABS_MAX = 10.0                      # les niveaux de PRBS restent dans [-X_ABS_MAX, X_ABS_MAX]
W_TRES_LOIN, E_TRES_LOIN = 0.2, 10.0  # prolongement du guidage lointain au-delà de E_LOIN (2 m) jusqu'à 10 m

BRUIT_ENTRAINEMENT = dict(sigma_x=SIGMA_G01, sigma_vx=SIGMA_G01, sigma_theta=SIGMA_G01, sigma_wy=SIGMA_G01)
PD_REF = (0.3, 0.45)                  # meilleur PD avec anticipation (reference_pd_x.py)
AGENT_PRECEDENT = RESULTATS / "agent_x_ppo"
N_ENVS = 16
PPO_KWARGS = dict(n_steps=512, batch_size=1024, n_epochs=10, learning_rate=3e-4, gamma=0.99, gae_lambda=0.95,
                  clip_range=0.2, ent_coef=0.0, policy_kwargs=dict(net_arch=[64, 64], log_std_init=-1.0))

# Scénarios de saturation FIXES pour l'évaluation (mêmes pour tous les agents comparés)
SCENARIOS_SAT = {
    "saut_5m": ((1.0, "p", 5.0),),
    "saut_8m": ((1.0, "p", 8.0),),
    "aller_retour_6m": ((1.0, "p", 6.0), (8.0, "p", -6.0)),
    "voiture_5.5m_0.3": ((1.0, "p", 5.5), (1.0, "v", 0.3)),     # cas de la mission LQ d'atterrissage
    "voiture_8m_0.8": ((1.0, "p", 8.0), (1.0, "v", 0.8)),
    "voiture_4m_0.5_freine": ((1.0, "p", 4.0), (1.0, "v", 0.5), (10.0, "v", 0.0)),
}


# ------------------------------------------------------------------ générateurs de consignes
def _bits_prbs(rng, n_bits=6):
    """Séquence de longueur maximale (registre à décalage x^6 + x^5 + 1), à partir d'un point aléatoire."""
    etat, seq = [1] * n_bits, []
    for _ in range(2 ** n_bits - 1):
        seq.append(etat[-1])
        etat = [etat[5] ^ etat[4]] + etat[:-1]
    k = int(rng.integers(0, len(seq)))
    return np.roll(np.array(seq), -k)


def mission_prbs(rng):
    """Bascule entre deux niveaux écartés de 2 à 8 m, aux instants d'une PRBS de période de bit Tb."""
    ecart = float(rng.uniform(*PRBS_ECART))
    bas = float(rng.uniform(-X_ABS_MAX + ecart, X_ABS_MAX - ecart)) if rng.random() < 0.5 else 0.0
    niveaux = (bas, bas + ecart) if rng.random() < 0.5 else (bas, bas - ecart)
    tb = float(rng.uniform(*PRBS_TB))
    bits = _bits_prbs(rng)
    t0 = float(rng.uniform(0.5, 2.0))
    ev, x_prec = [], 0.0
    for i, b in enumerate(bits):
        t = t0 + i * tb
        if t > N_STEPS * TS - 1.0:
            break
        cible = niveaux[int(b)]
        if cible != x_prec:
            ev.append((t, "p", cible - x_prec)); x_prec = cible
    return tuple(ev) if ev else ((t0, "p", niveaux[1]),)


def mission_poursuite(rng):
    """La cible est devant et roule ; une fois sur deux, elle change ensuite de vitesse (ou s'arrête)."""
    t0 = float(rng.uniform(0.5, 2.0))
    signe = 1.0 if rng.random() < 0.8 else -1.0
    d0 = signe * float(rng.uniform(*POURSUITE_D0))
    v = signe * float(rng.uniform(*POURSUITE_V))
    ev = [(t0, "p", d0), (t0, "v", v)]
    if rng.random() < 0.5:
        t1 = float(rng.uniform(t0 + 6.0, N_STEPS * TS - 3.0))
        ev.append((t1, "v", 0.0 if rng.random() < 0.4 else signe * float(rng.uniform(*POURSUITE_V))))
    return tuple(ev)


def recompense_sat(e, e_v, du):
    """Récompense de env_x (inchangée sous 2 m d'écart) + pente prolongée jusqu'à E_TRES_LOIN."""
    r = recompense(e, e_v, du)
    if abs(e) > E_LOIN:
        r -= W_TRES_LOIN * (min(abs(e), E_TRES_LOIN) - E_LOIN) / (E_TRES_LOIN - E_LOIN)
    return r


class XEnvSaturation(XEnv):
    """XEnv dont la consigne est tirée parmi les 3 familles, avec la récompense prolongée."""

    def __init__(self, consigne_imposee=None, **kw):
        super().__init__(**kw)
        self.consigne_imposee = consigne_imposee   # None : tirage parmi les 3 familles ; sinon consigne fixe (évaluation)
        self.rng_m = None

    def reset(self, seed=None, options=None):
        if self.consigne_imposee is not None:
            self.evenements_fixes = self.consigne_imposee
        else:
            if self.rng_m is None or seed is not None:
                self.rng_m = np.random.default_rng(seed)
            u = self.rng_m.random()
            f = mission_prbs if u < P_PRBS else (mission_poursuite if u < P_PRBS + P_POURSUITE else mission_aleatoire)
            self.evenements_fixes = f(self.rng_m)
        return super().reset(seed=seed, options=options)

    def step(self, action):
        a_avant = self.a_prev                      # action du pas précédent (XEnv.step la remplace)
        obs, _, term, trunc, info = super().step(action)
        x_ref, v_ref = consigne_x(self.k * TS, self.ev)
        du = self.a_prev - a_avant
        return obs, float(recompense_sat(x_ref - self.boucle.x, v_ref - self.boucle.vx, du)), term, trunc, info


# ------------------------------------------------------------------ évaluation
def politique_agent(model):
    return lambda obs, m, v_ref: float(np.asarray(model.predict(obs, deterministic=True)[0]).reshape(-1)[0])


def vol_sat(pol, ev, sigma=0.0, graine=0):
    """Un vol de 20 s sur une consigne fixe, avec la récompense prolongée. Renvoie trajectoire et indicateurs."""
    env = XEnvSaturation(consigne_imposee=ev, sigma_x=sigma, sigma_vx=sigma, sigma_theta=sigma, sigma_wy=sigma)
    obs, info = env.reset(seed=graine)
    T ={k: [] for k in ("t", "x", "x_ref", "a")}
    R = 0.0
    for _ in range(N_STEPS):
        m = info["m"]
        a = float(np.clip(pol(obs, m, consigne_x(m["t"], env.ev)[1]), -1.0, 1.0))
        obs, r, _, _, info = env.step([a])
        R += r
        for k, v in zip(T, (env.k * TS, env.boucle.x, consigne_x(env.k * TS, env.ev)[0], a)):
            T[k].append(v)
    T = {k: np.asarray(v) for k, v in T.items()}
    e = T["x_ref"] - T["x"]
    # fin du rattrapage : premier instant après lequel |e| reste sous 10 cm
    hors = np.where(np.abs(e) > 0.10)[0]
    t_rattrape = float(T["t"][hors[-1] + 1]) if len(hors) and hors[-1] + 1 < len(e) else (0.0 if not len(hors) else np.nan)
    depasse = float(np.max(-np.sign(ev[0][2]) * e))          # dépassement de la cible dans le sens du mouvement
    return T, dict(rendement=float(R), rmse=float(np.sqrt(np.mean(e ** 2))), t_rattrapage=t_rattrape,
                   depassement=max(depasse, 0.0), saturation=float(np.mean(np.abs(T["a"]) > 0.99)))


def eval_complete(pol, sigma=0.0):
    normales = [vol_sat(pol, ev, sigma, graine=i)[1] for i, ev in enumerate(missions_alea())]
    sat = {n: vol_sat(pol, ev, sigma)[1] for n, ev in SCENARIOS_SAT.items()}
    res = dict(normales_rendement=float(np.mean([r["rendement"] for r in normales])),
               normales_rmse=float(np.mean([r["rmse"] for r in normales])),
               sat_rendement=float(np.mean([r["rendement"] for r in sat.values()])),
               sat_rmse=float(np.mean([r["rmse"] for r in sat.values()])), sat=sat)
    res["score"] = res["normales_rendement"] + res["sat_rendement"]
    return res


ENTETE = f"{'':<22}{'score':>8} | {'normales : rend.':>17}{'RMSE':>8} | {'saturation : rend.':>19}{'RMSE':>8}"


def ligne(nom, r):
    return (f"{nom:<22}{r['score']:>8.1f} | {r['normales_rendement']:>17.1f}{r['normales_rmse']:>8.3f} | "
            f"{r['sat_rendement']:>19.1f}{r['sat_rmse']:>8.3f}")


class EvalCallback(BaseCallback):
    def __init__(self, freq, out_dir):
        super().__init__()
        self.freq, self.out_dir, self.next, self.best, self.history, self.t0 = freq, out_dir, freq, -np.inf, [], time.time()
        self.debut = 0

    def _on_training_start(self):
        self.debut = self.num_timesteps            # en reprise, num_timesteps ne repart pas de 0
        self.next = self.debut + self.freq

    def _on_step(self):
        if self.num_timesteps < self.next:
            return True
        self.next += self.freq
        r = eval_complete(politique_agent(self.model))
        r["pas"] = self.num_timesteps
        self.history.append({k: v for k, v in r.items() if k != "sat"})
        star = ""
        if r["score"] > self.best:
            self.best = r["score"]; self.model.save(self.out_dir / "best_model"); star = "  <- meilleur"
        vit = (self.num_timesteps - self.debut) / (time.time() - self.t0)
        print(ligne(f"{self.num_timesteps:>8}", r) + f"  ({vit:.0f}/s){star}", flush=True)
        return True


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=4_000_000)
    p.add_argument("--eval-freq", type=int, default=200_000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--depart", default=None,
                   help="dossier d'un agent à reprendre (ex. resultats/agent_x_ppo) ; par défaut : depuis zéro")
    a = p.parse_args()
    out = RESULTATS / time.strftime("agent_x_ppo_saturation_%Y%m%d_%H%M")
    out.mkdir(parents=True, exist_ok=True)
    print(f"Sorties dans {out}")

    pd = pd_anticipation(*PD_REF)
    refs = {f"PD {PD_REF[0]}/{PD_REF[1]}": pd}
    if (AGENT_PRECEDENT / "best_model.zip").exists():
        refs["agent précédent"] = politique_agent(PPO.load(AGENT_PRECEDENT / "best_model", device="cpu"))
    print(ENTETE)
    eval_refs = {}
    for nom, pol in refs.items():
        eval_refs[nom] = eval_complete(pol)
        print(ligne(nom, eval_refs[nom]))

    env = make_vec_env(XEnvSaturation, n_envs=N_ENVS, seed=a.seed, env_kwargs=BRUIT_ENTRAINEMENT, vec_env_cls=DummyVecEnv)
    if a.depart:
        dep = Path(a.depart)
        dep = dep if dep.is_absolute() else PYTHON / dep
        agent = PPO.load(dep / "best_model", env=env, device="cpu", tensorboard_log=str(out))
        print(f"Reprise de {dep}")
    else:
        agent = PPO("MlpPolicy", env, seed=a.seed, device="cpu", verbose=0, tensorboard_log=str(out), **PPO_KWARGS)
    cb = EvalCallback(a.eval_freq, out)
    t0 = time.time()
    agent.learn(total_timesteps=a.steps, callback=cb, tb_log_name="tensorboard", reset_num_timesteps=not a.depart)
    duree = time.time() - t0
    if not (out / "best_model.zip").exists():
        agent.save(out / "best_model")

    # ---- comparaison finale : saturation (scénarios fixes) + évaluation standard de reference_pd_x.py
    best = politique_agent(PPO.load(out / "best_model", device="cpu"))
    ctrls = {"nouvel agent": best, **refs}
    print(f"\n{a.steps} transitions en {duree / 60:.1f} min.\nScénarios de saturation (sans bruit) : "
          f"rendement / temps de rattrapage (|e| < 10 cm) / dépassement de la cible / saturation de theta_ref")
    print(f"{'':<26}" + "".join(f"{n:>24}" for n in ctrls))
    finales = {n: eval_complete(c) for n, c in ctrls.items()}
    for sc in SCENARIOS_SAT:
        cases = []
        for n in ctrls:
            s = finales[n]["sat"][sc]
            cases.append(f"{s['rendement']:6.1f} /{s['t_rattrapage']:5.1f}s /{s['depassement']:4.2f}m /{100 * s['saturation']:3.0f}%")
        print(f"{sc:<26}" + "".join(f"{c:>24}" for c in cases))
    print("\nÉvaluation standard (reference_pd_x.py, sans bruit puis bruit g ~ 0,1) :")
    standard = {}
    for n, c in ctrls.items():
        for sig in (0.0, SIGMA_G01):
            r, _ = evaluer(c, sig)
            standard[f"{n} | bruit {sig}"] = r
            al, s5 = r["aleatoire_moyenne"], r["saut_5m"]
            print(f"  {n:<20} bruit {sig:<6}: 20 missions {al['rendement']:6.1f} / RMSE {al['rmse']:.3f} m / agitation "
                  f"{al['agitation']:.4f} | saut 5 m {s5['rendement']:6.1f}, réponse {s5['reponse']:.2f} s | voiture "
                  f"écart final {100 * r['voiture_0.3']['e_fin']:.1f} cm")

    # ---- figures
    H = cb.history
    if H:
        fig, ax = plt.subplots(1, 2, figsize=(11, 4))
        for k, (cle, titre) in enumerate((("normales_rendement", "missions normales"), ("sat_rendement", "saturation"))):
            ax[k].plot([h["pas"] for h in H], [h[cle] for h in H], "o-", label="agent")
            for (n, r), st in zip(eval_refs.items(), ("--", ":")):
                ax[k].axhline(r[cle], color="k", ls=st, label=n)
            ax[k].set_xlabel("transitions"); ax[k].set_ylabel(f"rendement moyen ({titre})"); ax[k].legend(); ax[k].grid(alpha=.3)
        fig.tight_layout(); fig.savefig(out / "learning_curve.png", dpi=110); plt.close(fig)

    scen = ("saut_8m", "aller_retour_6m", "voiture_5.5m_0.3", "voiture_8m_0.8")
    fig, ax = plt.subplots(3, len(scen), figsize=(5 * len(scen), 9), sharex="col")
    for j, sc in enumerate(scen):
        for (n, c), st in zip(ctrls.items(), ("-", "--", ":")):
            T, _ = vol_sat(c, SCENARIOS_SAT[sc])
            ax[0, j].plot(T["t"], T["x"], st, label=n); ax[1, j].plot(T["t"], T["x_ref"] - T["x"], st, label=n)
            ax[2, j].plot(T["t"], THETA_MAX * T["a"], st, lw=1, label=n)
        ax[0, j].plot(T["t"], T["x_ref"], "k", lw=0.8, alpha=0.5, label="consigne lissée")
        ax[0, j].set_title(sc)
        for i, yl in enumerate(("x [m]", "erreur [m]", "θ_ref [rad]")):
            ax[i, j].set_ylabel(yl); ax[i, j].grid(alpha=.3)
        ax[2, j].set_xlabel("t [s]")
    ax[0, 0].legend(fontsize=8)
    fig.suptitle("Agent x entraîné aux saturations, contre l'agent précédent et le PD (sans bruit)")
    fig.tight_layout(); fig.savefig(out / "vols.png", dpi=110); plt.close(fig)

    json.dump({"reglages": dict(vars(a), n_envs=N_ENVS, ppo=PPO_KWARGS, bruit=BRUIT_ENTRAINEMENT, pd_ref=PD_REF,
                                p_prbs=P_PRBS, p_poursuite=P_POURSUITE, prbs_ecart=PRBS_ECART, prbs_tb=PRBS_TB,
                                poursuite_d0=POURSUITE_D0, poursuite_v=POURSUITE_V,
                                recompense_prolongee=dict(w=W_TRES_LOIN, e_max=E_TRES_LOIN)),
               "duree_min": duree / 60, "historique": H, "saturation": finales, "standard": standard},
              open(out / "results.json", "w", encoding="utf-8"), indent=2, default=str, ensure_ascii=False)
    print(f"Écrit dans {out}")


if __name__ == "__main__":
    main()
