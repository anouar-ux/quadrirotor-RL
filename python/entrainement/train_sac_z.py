"""
Étape 4 : entraînement d'un agent SAC sur DroneEnv (tenue d'une hauteur cible aléatoire), comparé au LQI.

    python train_sac_z.py                         # 100 000 pas, CPU (~20 min)
    python train_sac_z.py --steps 2000 --eval-freq 1000    # essai rapide
    tensorboard --logdir agents                # suivre l'apprentissage dans le navigateur

Chaque lancement crée un dossier resultats/agent_z_sac_AAAAMMJJ_HHMM/ contenant :
    best_model.zip      le meilleur agent (au sens de l'évaluation), rechargeable avec SAC.load
    policy.mat          l'acteur exporté pour MATLAB (rl_controller.m)        } à copier dans rl_interface/
    traj_model.mat      le vol de référence rejoué par EINT_RL_eval.m         } à l'étape 5
    results.json        réglages, historique des évaluations, résultats finaux agent / LQI
    learning_curve.png, flights.png
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # dossier python/ (chemins.py)
from chemins import MATLAB, PYTHON  # noqa: E402  (rend importables les sous-dossiers de python/)
HERE = PYTHON                # racine python/ : modeles/, resultats/ ; HERE.parent = EI_RL/
from env_z import N_STEPS, TS, DroneEnv, mission

sys.path.insert(0, str(HERE.parent / "rl_interface"))       # export_policy.py, fourni par le sujet (à ne pas modifier)
from export_policy import export_for_matlab                  # noqa: E402

# =============================================================================================
# 1. Réglages
# =============================================================================================
BRUIT_ENTRAINEMENT = dict(sigma_z=0.005, sigma_vz=0.005, sigma_az=0.005)   # ~ g_noise = 0.1 dans Simulink
EVAL_SEEDS = range(20)          # 20 hauteurs cibles FIXES (graines 0..19) : évaluations comparables entre elles
Z_EXPORT = 0.5                  # hauteur du vol exporté vers MATLAB : mission() de rl_controller.m devra la reproduire
GAINS_LQI = (0.93, 0.44, 0.395) # (Kp, Kd, Ki) de commande_LQ/gains_LQI.m divisés par 2 : stables à 50 ms (vérifié sur Simulink)
U0_LQI = 0.38                   # compensation de la gravité du LQI


# =============================================================================================
# 2. Les deux contrôleurs comparés, avec la même interface : reset(m) au début du vol, act(obs, m) -> u
# =============================================================================================
class AgentController:
    """La politique SAC. Elle ne voit que l'observation (calculée par l'env à partir des mesures)."""

    def __init__(self, agent):
        self.agent = agent

    def reset(self, m):
        pass                                                   # sa seule mémoire (l'intégrale) est dans l'observation

    def act(self, obs, m):
        action, _ = self.agent.predict(obs, deterministic=True)   # déterministe : pas d'exploration en évaluation
        return float(np.asarray(action).reshape(-1)[0])


class LQIController:
    """Référence classique : le LQI, gains réglés pour 50 ms. Il suit la MÊME consigne lissée que l'agent
    (m["z_ref"]), sans filtre supplémentaire, et ne lit que les mesures m, comme dans Simulink."""

    def __init__(self, gains=GAINS_LQI, u0=U0_LQI):
        self.kp, self.kd, self.ki = gains
        self.u0 = u0

    def reset(self, m):
        self.integ = 0.0

    def act(self, obs, m):
        e = m["z_ref"] - m["z"]
        self.integ += e * TS
        u = self.u0 + self.kp * e - self.kd * m["vz"] + self.ki * self.integ
        return float(np.clip(u, -1.0, 1.0))


# =============================================================================================
# 3. Évaluation : un vol, puis une série de vols sur les hauteurs fixes
# =============================================================================================
def flight(env, ctrl, seed):
    """Un vol complet de 10 s. Renvoie le rendement et les séries temporelles."""
    obs, info = env.reset(seed=seed)
    ctrl.reset(info["m"])
    total, z, u, truncated = 0.0, [], [], False
    while not truncated:
        a = ctrl.act(obs, info["m"])
        obs, r, _, truncated, info = env.step(np.array([a], dtype=np.float32))
        total += r
        z.append(env.z)
        u.append(a)
    return total, np.array(z), np.array(u), env.z_ref


def metrics(total, z, u, z_cible):
    """Indicateurs d'un vol. RMSE par rapport à la consigne lissée (celle de la récompense) et à l'échelon brut."""
    t = np.arange(1, len(z) + 1) * TS
    z_lisse = np.array([mission(x, z_cible) for x in t])
    sens = np.sign(z_cible) or 1.0                              # le drone part de 0 : on mesure le dépassement côté cible
    return dict(rendement=total,
                rmse_lisse=float(np.sqrt(np.mean((z_lisse - z) ** 2))),
                rmse_brut=float(np.sqrt(np.mean((z_cible - z) ** 2))),
                depassement=float(max(0.0, np.max(sens * (z - z_cible)))),
                erreur_finale=float(abs(z_cible - z[-1])),
                agitation_u=float(np.mean(np.abs(np.diff(u)))))   # variation moyenne de la commande d'un pas à l'autre


def evaluate(ctrl, seeds=EVAL_SEEDS):
    """Moyenne des indicateurs sur les hauteurs fixes, SANS bruit (comme l'export)."""
    env = DroneEnv()
    rows = [metrics(*flight(env, ctrl, s)) for s in seeds]
    return {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}


def ligne(nom, m):
    return (f"{nom:<14}{m['rendement']:>10.1f}{m['rmse_lisse']:>10.3f}{m['rmse_brut']:>10.3f}"
            f"{m['depassement']:>9.3f}{m['erreur_finale']:>9.3f}{m['agitation_u']:>10.3f}")


ENTETE = f"{'':<14}{'rendement':>10}{'RMSE liss':>10}{'RMSE brut':>10}{'dépass.':>9}{'err. fin':>9}{'agit. u':>10}"


# =============================================================================================
# 4. Callback : appelé par SB3 à chaque pas d'entraînement ; évalue l'agent tous les `eval_freq` pas
# =============================================================================================
class EvalCallback(BaseCallback):
    def __init__(self, eval_freq, out_dir, ref_lqi):
        super().__init__()
        self.eval_freq, self.out_dir, self.ref = eval_freq, out_dir, ref_lqi
        self.best, self.history, self.t0 = -np.inf, [], time.time()
        self.next_eval = eval_freq

    def _on_step(self):
        # num_timesteps compte les transitions de TOUS les drones (plusieurs par appel avec un VecEnv, ex. PPO)
        if self.num_timesteps < self.next_eval:
            return True                                            # True : continuer l'entraînement
        self.next_eval += self.eval_freq
        res = evaluate(AgentController(self.model))
        res["pas"] = self.num_timesteps
        self.history.append(res)
        for k in ("rendement", "rmse_lisse", "rmse_brut", "agitation_u"):    # courbes TensorBoard
            self.logger.record(f"eval/{k}", res[k])
        star = ""
        if res["rendement"] > self.best:                           # critère : le rendement, c'est ce que SAC optimise
            self.best = res["rendement"]
            self.model.save(self.out_dir / "best_model")
            star = "  <- meilleur, sauvegardé"
        vitesse = self.num_timesteps / (time.time() - self.t0)
        print(ligne(f"{self.num_timesteps:>7} pas", res) + f"   ({vitesse:.0f} pas/s){star}", flush=True)
        return True


# =============================================================================================
# 5. Figures
# =============================================================================================
def figures(history, ref, agent_ctrl, out_dir, nom_agent="agent SAC"):
    h = {k: [r[k] for r in history] for k in history[0]}
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.5))
    a1.plot(h["pas"], h["rendement"], "o-", label=nom_agent)
    a1.axhline(ref["rendement"], color="C1", ls="--", label="LQI (référence)")
    a1.set_xlabel("pas d'entraînement"); a1.set_ylabel("rendement moyen (20 vols)"); a1.legend(); a1.grid(alpha=.3)
    a2.plot(h["pas"], h["rmse_lisse"], "o-", label=f"{nom_agent}, RMSE / consigne lissée")
    a2.axhline(ref["rmse_lisse"], color="C1", ls="--", label="LQI, RMSE / consigne lissée")
    a2.set_xlabel("pas d'entraînement"); a2.set_ylabel("RMSE [m]"); a2.legend(); a2.grid(alpha=.3)
    fig.suptitle("Évaluation sans bruit sur 20 hauteurs cibles fixes")
    fig.tight_layout(); fig.savefig(out_dir / "learning_curve.png", dpi=120)

    t = np.arange(1, N_STEPS + 1) * TS
    fig, axes = plt.subplots(2, 3, figsize=(15, 7), sharex=True)
    for j, cible in enumerate((0.5, 1.0, -0.5)):
        env = DroneEnv(z_ref_fixe=cible)
        axes[0, j].plot(t, [mission(x, cible) for x in t], "k--", lw=1, label="consigne lissée")
        for nom, ctrl, c in ((nom_agent, agent_ctrl, "C0"), ("LQI", LQIController(), "C1")):
            _, z, u, _ = flight(env, ctrl, seed=0)
            axes[0, j].plot(t, z, color=c, label=nom)
            axes[1, j].step(t, u, where="post", color=c, label=nom)
        axes[0, j].set_title(f"hauteur cible {cible:+.1f} m"); axes[0, j].grid(alpha=.3)
        axes[1, j].set_xlabel("t [s]"); axes[1, j].grid(alpha=.3)
    axes[0, 0].set_ylabel("z [m]"); axes[1, 0].set_ylabel("u"); axes[0, 0].legend(); axes[1, 0].legend()
    fig.tight_layout(); fig.savefig(out_dir / "flights.png", dpi=120)


# =============================================================================================
# 6. Programme principal
# =============================================================================================
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=100_000, help="nombre total de pas d'entraînement")
    p.add_argument("--eval-freq", type=int, default=10_000, help="évaluation tous les N pas")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cpu", help="cpu (recommandé pour des MLP) ou cuda")
    a = p.parse_args()

    out_dir = HERE / "resultats" / time.strftime("agent_z_sac_%Y%m%d_%H%M")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Sorties dans {out_dir}")

    # --- référence LQI, calculée une fois sur les mêmes 20 vols que l'agent
    ref = evaluate(LQIController())
    print("\n" + ENTETE)
    print(ligne("LQI (réf.)", ref))

    # --- l'agent : SAC, réglages par défaut de SB3 (2 réseaux 2x256, entropie automatique)
    env = DroneEnv(**BRUIT_ENTRAINEMENT)                       # entraînement AVEC bruit de mesure
    agent = SAC("MlpPolicy", env, seed=a.seed, device=a.device, verbose=0,
                ent_coef="auto_0.1",                           # entropie automatique, mais partant de 0.1 (et non 1)
                tensorboard_log=str(out_dir))                  # journaux TensorBoard dans out_dir/tensorboard_1
    callback = EvalCallback(a.eval_freq, out_dir, ref)
    t0 = time.time()
    agent.learn(total_timesteps=a.steps, callback=callback, tb_log_name="tensorboard")
    duree = time.time() - t0

    # --- le meilleur agent : bilan, export MATLAB, figures
    best = SAC.load(out_dir / "best_model", device=a.device)
    final = evaluate(AgentController(best))
    print(f"\nEntraînement : {a.steps} pas en {duree / 60:.1f} min. Meilleur agent (rendement {callback.best:.1f}) :")
    print(ENTETE)
    print(ligne("agent SAC", final))
    print(ligne("LQI (réf.)", ref))

    # export_for_matlab écrit policy.mat et traj_model.mat dans le dossier COURANT : on se place dans out_dir
    cwd = os.getcwd()
    os.chdir(out_dir)
    try:
        export_for_matlab(best, DroneEnv(z_ref_fixe=Z_EXPORT))   # SANS bruit, consigne fixe (reproduite par mission())
    finally:
        os.chdir(cwd)

    figures(callback.history, ref, AgentController(best), out_dir)
    json.dump({"reglages": dict(vars(a), bruit=BRUIT_ENTRAINEMENT, gains_lqi=GAINS_LQI, z_export=Z_EXPORT),
               "duree_min": duree / 60, "historique": callback.history, "agent_final": final, "lqi": ref},
              open(out_dir / "results.json", "w"), indent=2)
    print(f"\nÉcrit dans {out_dir} : best_model.zip, policy.mat, traj_model.mat, results.json, 2 figures")
    plt.show()


if __name__ == "__main__":
    main()
