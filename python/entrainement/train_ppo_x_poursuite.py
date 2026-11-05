"""
Entraînement PPO de l'agent de position x pour la POURSUITE d'une voiture rapide (jusqu'à 2,5 m/s) sur PoursuiteEnv.
Même méthode et mêmes réglages PPO que train_ppo_x.py ; ce qui change : l'environnement (référence de la mission, drone parfois
lancé à grande vitesse, observation en repère relatif, récompense avec courbe de freinage) et un CURRICULUM sur la vitesse de
la voiture (de 1 m/s à 2,5 m/s sur la première moitié de l'entraînement) : PPO apprend d'abord à suivre, puis à freiner.

    python python/entrainement/train_ppo_x_poursuite.py                      # 2 000 000 de transitions
    python python/entrainement/train_ppo_x_poursuite.py --steps 300000       # essai rapide
    python python/entrainement/train_ppo_x_poursuite.py --sans-courbe        # ablation : récompense de vitesse d'origine

Sélection du meilleur agent : score = -(instant de pose moyen + 10 x dépassement moyen), sur 40 vols aléatoires FIXES à
2,5 m/s avec le bruit de g_noise = 0,1 (le critère de la mission, pas la récompense). Sorties : resultats/agent_x_poursuite_*/.
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
from chemins import RESULTATS  # noqa: E402  (rend importables les sous-dossiers de python/)
from env_x_poursuite import PoursuiteEnv, V_CAR_MAX  # noqa: E402
from poursuite_x import (AgentNouveau, SIGMA_G01, afficher, charger_controleurs, evaluer, figure,  # noqa: E402
                         resume, scenarios_aleatoires, vol)

BRUIT_ENTRAINEMENT = dict(sigma_x=SIGMA_G01, sigma_vx=SIGMA_G01, sigma_theta=SIGMA_G01, sigma_wy=SIGMA_G01)  # g_noise 0,1
N_ENVS = 16
N_EVAL = 40
T_PENALITE = 25.0            # instant de pose compté pour un vol qui ne se pose pas [s] (le vol dure 20 s)
PPO_KWARGS = dict(n_steps=512, batch_size=1024, n_epochs=10, learning_rate=3e-4, gamma=0.99, gae_lambda=0.95,
                  clip_range=0.2, ent_coef=0.0, policy_kwargs=dict(net_arch=[64, 64], log_std_init=-1.0))


def score_rapide(model, g_x):
    ctl = AgentNouveau(model, g_x)
    rows = [vol(ctl, sc, SIGMA_G01, graine=i)[1] for i, sc in enumerate(scenarios_aleatoires(N_EVAL))]
    t = np.mean([r["t_pose"] if r["pose"] else T_PENALITE for r in rows])
    d = np.nanmean([r["depassement"] for r in rows])
    return -(t + 10 * d), resume(rows)


class Curriculum(BaseCallback):
    """Monte la vitesse maximale de la voiture de v_debut à V_CAR_MAX sur la fraction `frac` de l'entraînement."""

    def __init__(self, total, v_debut=1.0, frac=0.5):
        super().__init__()
        self.total, self.v_debut, self.frac = total, v_debut, frac

    def _vmax(self):
        return self.v_debut + (V_CAR_MAX - self.v_debut) * min(1.0, self.num_timesteps / (self.frac * self.total))

    def _on_rollout_start(self):
        self.training_env.env_method("set_vmax", self._vmax())

    def _on_step(self):
        return True


class EvalCallback(BaseCallback):
    def __init__(self, freq, out_dir, g_x, curriculum):
        super().__init__()
        self.freq, self.out_dir, self.g_x, self.cur = freq, out_dir, g_x, curriculum
        self.next, self.best, self.history, self.t0 = freq, -np.inf, [], time.time()

    def _on_step(self):
        if self.num_timesteps < self.next:
            return True
        self.next += self.freq
        s, r = score_rapide(self.model, self.g_x)
        r.update(pas=self.num_timesteps, score=s, v_max=self.cur._vmax())
        self.history.append(r)
        star = ""
        if s > self.best:
            self.best = s; self.model.save(self.out_dir / "best_model"); star = "  <- meilleur"
        print(f"{self.num_timesteps:>9}  v_max {r['v_max']:.2f}  pose {r['taux_pose']:>4.0%}  t_pose {r['t_pose_med']:>5.1f} s  "
              f"dépass. {100 * r['depassement_moy']:>5.1f} cm (max {100 * r['depassement_max']:>5.1f})  score {s:>7.2f}  "
              f"({self.num_timesteps / (time.time() - self.t0):.0f}/s){star}", flush=True)
        return True


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=2_000_000)
    p.add_argument("--eval-freq", type=int, default=100_000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--sans-courbe", action="store_true", help="ablation : suivi de la vitesse de consigne au lieu de la courbe de freinage")
    p.add_argument("--nom", default="agent_x_poursuite")
    a = p.parse_args()
    out = RESULTATS / time.strftime(f"{a.nom}_%Y%m%d_%H%M")
    out.mkdir(parents=True, exist_ok=True)
    print(f"Sorties dans {out}")

    env = make_vec_env(PoursuiteEnv, n_envs=N_ENVS, seed=a.seed, vec_env_cls=DummyVecEnv,
                       env_kwargs=dict(BRUIT_ENTRAINEMENT, courbe=not a.sans_courbe))
    agent = PPO("MlpPolicy", env, seed=a.seed, device="cpu", verbose=0, **PPO_KWARGS)
    cur = Curriculum(a.steps)
    cb = EvalCallback(a.eval_freq, out, PoursuiteEnv().g_x, cur)
    t0 = time.time()
    agent.learn(total_timesteps=a.steps, callback=[cur, cb])
    duree = time.time() - t0
    agent.save(out / "final_model")

    print(f"\n{a.steps} transitions en {duree / 60:.1f} min. Comparaison complète (poursuite_x.py), bruit g_noise = 0,1 :")
    res, traj = evaluer(charger_controleurs(out))
    afficher(res)
    figure(traj, out / "poursuite_comparaison.png")

    H = cb.history
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    pas = [h["pas"] for h in H]
    ax[0].plot(pas, [h["taux_pose"] for h in H], "o-"); ax[0].set_ylabel("taux de pose (40 vols à 2,5 m/s)")
    ax[1].plot(pas, [100 * h["depassement_moy"] for h in H], "o-", label="moyen")
    ax[1].plot(pas, [100 * h["depassement_max"] for h in H], "s--", label="max"); ax[1].set_ylabel("dépassement [cm]"); ax[1].legend()
    ax[2].plot(pas, [h["v_max"] for h in H], "o-"); ax[2].set_ylabel("vitesse max de la voiture (curriculum) [m/s]")
    for x in ax:
        x.set_xlabel("transitions"); x.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(out / "learning_curve.png", dpi=110)
    json.dump({"reglages": dict(vars(a), n_envs=N_ENVS, ppo=PPO_KWARGS, bruit=BRUIT_ENTRAINEMENT), "duree_min": duree / 60,
               "historique": H, "comparaison": res}, open(out / "results.json", "w"), indent=2, default=str)
    print(f"Écrit dans {out}")


if __name__ == "__main__":
    main()
