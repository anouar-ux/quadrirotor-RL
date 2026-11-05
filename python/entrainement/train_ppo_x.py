"""
Étape C2.3 : entraînement PPO de l'agent de POSITION x sur XEnv (16 drones en parallèle), même méthode que z et lacet.
L'agent donne theta_ref = 0,2 a ; le LQ d'attitude (boucle interne) le suit.

    python python/entrainement/train_ppo_x.py                    # 1 000 000 de transitions
    python python/entrainement/train_ppo_x.py --steps 200000     # essai rapide

Évaluation (sans bruit, déterministe) sur les scénarios de reference_pd_x.py : 20 missions aléatoires fixes, sauts de 2 m et
5 m, cible qui roule ; comparée au meilleur PD avec anticipation (référence C2.2).
Sorties : resultats/agent_x_ppo_AAAAMMJJ_HHMM/ (best_model.zip, results.json, figures).
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
from reference_pd_x import SIGMA_G01, evaluer, missions_alea, pd_anticipation, vol  # noqa: E402
from env_x import THETA_MAX, XEnv  # noqa: E402

BRUIT_ENTRAINEMENT = dict(sigma_x=SIGMA_G01, sigma_vx=SIGMA_G01, sigma_theta=SIGMA_G01, sigma_wy=SIGMA_G01)  # g_noise 0,1
PD_REF = (0.3, 0.45)                 # meilleur PD avec anticipation (reference_pd_x.py, C2.2)
N_ENVS = 16
PPO_KWARGS = dict(n_steps=512, batch_size=1024, n_epochs=10, learning_rate=3e-4, gamma=0.99, gae_lambda=0.95,
                  clip_range=0.2, ent_coef=0.0, policy_kwargs=dict(net_arch=[64, 64], log_std_init=-1.0))


def politique_agent(model):
    return lambda obs, m, v_ref: float(np.asarray(model.predict(obs, deterministic=True)[0]).reshape(-1)[0])


def eval_rapide(pol):
    """Pour le suivi de l'apprentissage : 20 missions aléatoires + saut de 5 m (saturé), sans bruit."""
    rows = [vol(pol, ev, 0.0, graine=i)[1] for i, ev in enumerate(missions_alea())]
    res = {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}
    s5 = vol(pol, ((1.0, "p", 5.0),))[1]
    res.update(saut5_rendement=s5["rendement"], saut5_rmse=s5["rmse"])
    return res


ENTETE = f"{'':<16}{'rendement':>10}{'RMSE':>9}{'écart max':>10}{'agit.':>8} | {'saut 5 m : rend.':>17}{'RMSE':>8}"


def ligne(nom, r):
    return (f"{nom:<16}{r['rendement']:>10.1f}{r['rmse']:>9.4f}{r['e_max']:>10.3f}{r['agitation']:>8.4f} | "
            f"{r['saut5_rendement']:>17.1f}{r['saut5_rmse']:>8.3f}")


class EvalCallback(BaseCallback):
    def __init__(self, freq, out_dir):
        super().__init__()
        self.freq, self.out_dir, self.next, self.best, self.history, self.t0 = freq, out_dir, freq, -np.inf, [], time.time()

    def _on_step(self):
        if self.num_timesteps < self.next:
            return True
        self.next += self.freq
        r = eval_rapide(politique_agent(self.model)); r["pas"] = self.num_timesteps
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
    out = RESULTATS / time.strftime("agent_x_ppo_%Y%m%d_%H%M")
    out.mkdir(parents=True, exist_ok=True)
    print(f"Sorties dans {out}")
    pd = pd_anticipation(*PD_REF)
    ref = eval_rapide(pd)
    print(ENTETE); print(ligne(f"PD {PD_REF[0]}/{PD_REF[1]} (réf.)", ref))

    env = make_vec_env(XEnv, n_envs=N_ENVS, seed=a.seed, env_kwargs=BRUIT_ENTRAINEMENT, vec_env_cls=DummyVecEnv)
    agent = PPO("MlpPolicy", env, seed=a.seed, device="cpu", verbose=0, tensorboard_log=str(out), **PPO_KWARGS)
    cb = EvalCallback(a.eval_freq, out)
    t0 = time.time()
    agent.learn(total_timesteps=a.steps, callback=cb, tb_log_name="tensorboard")
    duree = time.time() - t0

    best = PPO.load(out / "best_model", device="cpu")
    pol = politique_agent(best)
    print(f"\n{a.steps} transitions en {duree / 60:.1f} min. Comparaison complète (reference_pd_x.py) :")
    comp, trajs = {}, {}
    for nom, c in (("agent PPO", pol), (f"PD {PD_REF[0]}/{PD_REF[1]}", pd)):
        for sig in (0.0, SIGMA_G01):
            res, traj = evaluer(c, sig)
            comp[f"{nom} | bruit {sig}"] = res
            if sig == 0.0:
                trajs[nom] = traj
    print(f"{'':<28}{'saut 2 m':>22}{'saut 5 m':>22}{'voiture 0,3':>14}{'cible 1,0':>12}{'20 missions':>22}")
    print(f"{'':<28}{'rend. / dép. / rép.':>22}{'rend. / dép. / rép.':>22}{'écart fin':>14}{'écart fin':>12}"
          f"{'rend. / RMSE / agit.':>22}")
    for cle, r in comp.items():
        s2, s5, al = r["saut_2m"], r["saut_5m"], r["aleatoire_moyenne"]
        print(f"{cle:<28}{s2['rendement']:>8.1f} /{s2['depassement']:>5.1f} % /{s2['reponse']:>5.2f}"
              f"{s5['rendement']:>8.1f} /{s5['depassement']:>5.1f} % /{s5['reponse']:>5.2f}"
              f"{100 * r['voiture_0.3']['e_fin']:>11.1f} cm{100 * r['cible_1.0']['e_fin']:>9.1f} cm"
              f"{al['rendement']:>9.1f} /{al['rmse']:>6.3f} /{al['agitation']:>6.4f}")

    # figures : apprentissage + scénarios (agent contre PD)
    H = cb.history
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot([h["pas"] for h in H], [h["rendement"] for h in H], "o-", label="agent")
    ax[0].axhline(ref["rendement"], color="k", ls="--", label="PD (réf.)")
    ax[0].set_xlabel("transitions"); ax[0].set_ylabel("rendement moyen (20 missions)"); ax[0].legend(); ax[0].grid(alpha=.3)
    ax[1].plot([h["pas"] for h in H], [h["rmse"] for h in H], "o-", label="agent")
    ax[1].axhline(ref["rmse"], color="k", ls="--", label="PD (réf.)")
    ax[1].set_xlabel("transitions"); ax[1].set_ylabel("RMSE x [m]"); ax[1].set_yscale("log"); ax[1].legend(); ax[1].grid(alpha=.3)
    fig.tight_layout(); fig.savefig(out / "learning_curve.png", dpi=110)

    fig, ax = plt.subplots(3, 3, figsize=(16, 9), sharex="col")
    titres = {"saut_2m": "Saut de 2 m", "saut_5m": "Saut de 5 m (θ saturé)", "voiture_0.3": "Cible qui roule à 0,3 m/s"}
    for j, sc in enumerate(titres):
        for (nom, traj), st in zip(trajs.items(), ("-", "--")):
            T = traj[sc]
            ax[0, j].plot(T["t"], T["x"], st, label=nom); ax[1, j].plot(T["t"], T["x_ref"] - T["x"], st, label=nom)
            ax[2, j].plot(T["t"], THETA_MAX * T["a"], st, label=nom)
        T = next(iter(trajs.values()))[sc]
        ax[0, j].plot(T["t"], T["x_ref"], "k:", label="consigne lissée")
        ax[0, j].set_title(titres[sc])
        for i, yl in enumerate(("x [m]", "erreur [m]", "θ_ref [rad]")):
            ax[i, j].set_ylabel(yl); ax[i, j].grid(alpha=.3)
        ax[2, j].set_xlabel("t [s]")
    ax[0, 0].legend(fontsize=8)
    fig.suptitle("Agent PPO x contre la référence PD avec anticipation (sans bruit)")
    fig.tight_layout(); fig.savefig(out / "vols.png", dpi=110)
    json.dump({"reglages": dict(vars(a), n_envs=N_ENVS, ppo=PPO_KWARGS, bruit=BRUIT_ENTRAINEMENT, pd_ref=PD_REF),
               "duree_min": duree / 60, "historique": H, "comparaison": comp}, open(out / "results.json", "w"),
              indent=2, default=str)
    print(f"Écrit dans {out}")


if __name__ == "__main__":
    main()
