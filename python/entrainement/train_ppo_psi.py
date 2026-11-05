"""
Étape B2 : entraînement PPO de l'agent de LACET sur YawEnv (16 drones en parallèle), même méthode que train_ppo_z.py (z).

    python train_ppo_psi.py                      # 1 000 000 de transitions (~4 min)
    python train_ppo_psi.py --steps 200000       # essai rapide

Évaluation (sans bruit, déterministe) sur 20 missions aléatoires FIXES + la mission du scan, comparée au LQI de lacet.
Sorties : resultats/agent_psi_ppo_AAAAMMJJ_HHMM/ (best_model.zip, results.json, figures).
"""
import sys
import argparse
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # dossier python/ (chemins.py)
from chemins import PYTHON  # noqa: E402  (rend importables les sous-dossiers de python/)
HERE = PYTHON                # racine python/ : modeles/, resultats/ ; HERE.parent = EI_RL/
from env_psi import MISSION_SCAN, N_STEPS, TS, U_PSI_MAX, YawEnv, consigne_lacet

BRUIT_ENTRAINEMENT = dict(sigma_psi=0.005, sigma_wz=0.005, sigma_wdz=0.05)   # ~ g_noise = 0.1
EVAL_SEEDS = range(20)
GAINS_LQI_PSI = (0.252011, 0.124877, 0.097409)   # ident_psi.m (Kp, Kd, Ki), en unités de u_psi
N_ENVS = 16
PPO_KWARGS = dict(n_steps=512, batch_size=1024, n_epochs=10, learning_rate=3e-4, gamma=0.99, gae_lambda=0.95,
                  clip_range=0.2, ent_coef=0.0, policy_kwargs=dict(net_arch=[64, 64], log_std_init=-1.0))


class Agent:
    def __init__(self, model):
        self.model = model

    def reset(self):
        pass

    def act(self, obs, m):
        return float(np.asarray(self.model.predict(obs, deterministic=True)[0]).reshape(-1)[0])


class LQIPsi:
    def __init__(self, gains=GAINS_LQI_PSI):
        self.kp, self.kd, self.ki = gains

    def reset(self):
        self.integ = 0.0

    def act(self, obs, m):
        e = m["psi_ref"] - m["psi"]
        self.integ += e * TS
        return float(np.clip((self.kp * e - self.kd * m["wz"] + self.ki * self.integ) / U_PSI_MAX, -1, 1))


def vol(ctrl, sauts=None, seed=0, bruit=None):
    env = YawEnv(sauts_fixes=sauts, **(bruit or {}))
    obs, info = env.reset(seed=seed)
    ctrl.reset()
    total, psi, ref, a_ = 0.0, [], [], []
    for k in range(N_STEPS):
        a = ctrl.act(obs, info["m"])
        obs, r, _, _, info = env.step([a])
        total += r
        psi.append(env.psi); a_.append(a); ref.append(consigne_lacet((k + 1) * TS, env.sauts)[0])
    psi, ref, a_ = map(np.array, (psi, ref, a_))
    return dict(rendement=total, rmse=float(np.sqrt(np.mean((ref - psi) ** 2))), err_max=float(np.max(np.abs(ref - psi))),
                err_fin=float(abs(ref[-1] - psi[-1])), agitation=float(np.mean(np.abs(np.diff(a_))))), (psi, ref, a_)


def evaluer(ctrl):
    rows = [vol(ctrl, seed=s)[0] for s in EVAL_SEEDS]
    res = {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}
    scan = vol(ctrl, MISSION_SCAN)[0]
    res.update({f"scan_{k}": v for k, v in scan.items()})
    return res


ENTETE = f"{'':<14}{'rendement':>10}{'RMSE':>9}{'err max':>9}{'agit.':>8} | {'scan : RMSE':>12}{'err max':>9}{'err fin':>9}"


def ligne(nom, r):
    return (f"{nom:<14}{r['rendement']:>10.1f}{r['rmse']:>9.4f}{r['err_max']:>9.3f}{r['agitation']:>8.3f} | "
            f"{r['scan_rmse']:>12.4f}{r['scan_err_max']:>9.3f}{r['scan_err_fin']:>9.4f}")


class EvalCallback(BaseCallback):
    def __init__(self, freq, out_dir):
        super().__init__()
        self.freq, self.out_dir, self.next, self.best, self.history, self.t0 = freq, out_dir, freq, -np.inf, [], time.time()

    def _on_step(self):
        if self.num_timesteps < self.next:
            return True
        self.next += self.freq
        r = evaluer(Agent(self.model)); r["pas"] = self.num_timesteps
        self.history.append(r)
        star = ""
        if r["rendement"] > self.best:
            self.best = r["rendement"]; self.model.save(self.out_dir / "best_model"); star = "  <- meilleur"
        print(ligne(f"{self.num_timesteps:>8}", r) + f"  ({self.num_timesteps / (time.time() - self.t0):.0f}/s){star}", flush=True)
        return True


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=1_000_000)
    p.add_argument("--eval-freq", type=int, default=100_000)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    out = HERE / "resultats" / time.strftime("agent_psi_ppo_%Y%m%d_%H%M")
    out.mkdir(parents=True, exist_ok=True)
    print(f"Sorties dans {out}")
    ref = evaluer(LQIPsi())
    print(ENTETE); print(ligne("LQI (réf.)", ref))

    env = make_vec_env(YawEnv, n_envs=N_ENVS, seed=a.seed, env_kwargs=BRUIT_ENTRAINEMENT, vec_env_cls=DummyVecEnv)
    agent = PPO("MlpPolicy", env, seed=a.seed, device="cpu", verbose=0, tensorboard_log=str(out), **PPO_KWARGS)
    cb = EvalCallback(a.eval_freq, out)
    t0 = time.time()
    agent.learn(total_timesteps=a.steps, callback=cb, tb_log_name="tensorboard")
    duree = time.time() - t0

    best = PPO.load(out / "best_model", device="cpu")
    final = evaluer(Agent(best))
    print(f"\n{a.steps} transitions en {duree / 60:.1f} min. Meilleur agent :")
    print(ENTETE); print(ligne("agent PPO", final)); print(ligne("LQI (réf.)", ref))
    bruit = {n: [vol(c, MISSION_SCAN, seed=s, bruit=BRUIT_ENTRAINEMENT)[0] for s in range(5)]
             for n, c in (("agent", Agent(best)), ("lqi", LQIPsi()))}
    for n, rows in bruit.items():
        print(f"scan avec bruit, {n:<5}: RMSE {np.mean([r['rmse'] for r in rows]):.4f} rad, "
              f"err max {np.mean([r['err_max'] for r in rows]):.3f}, agitation {np.mean([r['agitation'] for r in rows]):.3f}")

    # figures : apprentissage + scan (agent contre LQI) + 3 missions aléatoires
    H = cb.history
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot([h["pas"] for h in H], [h["rendement"] for h in H], "o-", label="agent"); ax[0].axhline(ref["rendement"], color="k", ls="--", label="LQI")
    ax[0].set_xlabel("transitions"); ax[0].set_ylabel("rendement moyen"); ax[0].legend(); ax[0].grid(alpha=.3)
    ax[1].plot([h["pas"] for h in H], [h["rmse"] for h in H], "o-", label="agent"); ax[1].axhline(ref["rmse"], color="k", ls="--", label="LQI")
    ax[1].set_xlabel("transitions"); ax[1].set_ylabel("RMSE psi [rad]"); ax[1].set_yscale("log"); ax[1].legend(); ax[1].grid(alpha=.3)
    fig.tight_layout(); fig.savefig(out / "learning_curve.png", dpi=110)

    t = np.arange(1, N_STEPS + 1) * TS
    fig, ax = plt.subplots(4, 2, figsize=(12, 10), sharex=True)
    for col, (titre, sauts, seed) in enumerate((("Scan : un tour complet", MISSION_SCAN, 0), ("Mission aléatoire (graine 3)", None, 3))):
        for nom, c, st in (("agent PPO", Agent(best), "-"), ("LQI", LQIPsi(), "--")):
            _, (psi, r, a_) = vol(c, sauts, seed=seed)
            if nom == "agent PPO":
                ax[0, col].plot(t, r, "k:", lw=1, label="consigne")
            ax[0, col].plot(t, psi, st, label=nom); ax[1, col].plot(t, r - psi, st, label=nom)
            ax[2, col].step(t, a_ * U_PSI_MAX, st, where="post", label=nom)
        _, (psi, r, a_) = vol(Agent(best), sauts, seed=seed, bruit=BRUIT_ENTRAINEMENT)
        ax[3, col].plot(t, r - psi, label="agent, bruit g_noise ~0.1"); ax[3, col].step(t, a_ * U_PSI_MAX * 20, where="post", alpha=.5, label="u_psi x 20")
        ax[0, col].set_title(titre)
        for i, yl in enumerate(("psi [rad]", "erreur [rad]", "u_psi", "avec bruit")):
            ax[i, col].set_ylabel(yl); ax[i, col].grid(alpha=.3); ax[i, col].legend(fontsize=7)
    ax[3, 0].set_xlabel("t [s]"); ax[3, 1].set_xlabel("t [s]")
    fig.tight_layout(); fig.savefig(out / "vols.png", dpi=110)
    json.dump({"reglages": dict(vars(a), n_envs=N_ENVS, ppo=PPO_KWARGS, bruit=BRUIT_ENTRAINEMENT), "duree_min": duree / 60,
               "historique": H, "agent": final, "lqi": ref, "bruit_scan": bruit}, open(out / "results.json", "w"), indent=2, default=str)
    print(f"Écrit dans {out}")


if __name__ == "__main__":
    main()
