"""
Étape 4 (version PPO) : entraînement d'un agent PPO sur DroneEnv avec 16 drones simulés en parallèle.

    python train_ppo_z.py                                  # 2 000 000 de transitions (~10 min sur CPU)
    python train_ppo_z.py --steps 200000 --eval-freq 100000    # essai rapide
    tensorboard --logdir agents

Pourquoi PPO ici : SAC plafonne à ~70 mises à jour/s (coût fixe par mise à jour, GPU ou pas), alors que PPO
traite ~5 000 transitions/s avec 16 drones en parallèle sur CPU. PPO a besoin de plus d'expérience, mais
l'obtient beaucoup plus vite. export_for_matlab gère PPO (sortie « clip »), rl_controller.m aussi.

Mêmes sorties que train_sac_z.py, dans resultats/agent_z_ppo_AAAAMMJJ_HHMM/ (évaluation, comparaison au LQI, export, figures).
"""
from pathlib import Path
import sys
import argparse
import json
import os
import time

import matplotlib.pyplot as plt
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # dossier python/ (chemins.py)
from chemins import MATLAB, PYTHON  # noqa: E402  (rend importables les sous-dossiers de python/)
HERE = PYTHON                # racine python/ : modeles/, resultats/ ; HERE.parent = EI_RL/
from env_z import DroneEnv
from train_sac_z import (BRUIT_ENTRAINEMENT, ENTETE, HERE, Z_EXPORT, AgentController, EvalCallback, LQIController,
                       evaluate, export_for_matlab, figures, ligne)

# Réglages de PPO (ceux qui diffèrent des valeurs par défaut de SB3 sont commentés)
N_ENVS = 16                     # drones simulés en parallèle
U_EQ = 0.39                     # commande de vol stationnaire (étape 1) : point de départ de la politique
PPO_KWARGS = dict(
    n_steps=256,                # pas par drone avant chaque mise à jour -> 16 x 256 = 4096 transitions par lot
    batch_size=512,             # taille des mini-lots de la descente de gradient
    n_epochs=10,                # passages sur chaque lot
    learning_rate=3e-4,
    gamma=0.99,                 # horizon effectif ~ 100 pas = 5 s
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.0,               # pas de bonus d'entropie : l'exploration vient de l'écart-type appris
    policy_kwargs=dict(net_arch=[64, 64],      # réseau tanh 64 x 64 (défaut PPO), exporté tel quel vers MATLAB
                       log_std_init=-1.0),     # écart-type initial de l'exploration ~0.37 (au lieu de 1 : u est dans [-1, 1])
)
# Pas de VecNormalize : la normalisation serait cachée dans SB3 et rl_controller.m ne la reproduirait pas


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=2_000_000, help="nombre total de transitions (tous drones confondus)")
    p.add_argument("--eval-freq", type=int, default=100_000, help="évaluation toutes les N transitions")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()

    out_dir = HERE / "resultats" / time.strftime("agent_z_ppo_%Y%m%d_%H%M")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Sorties dans {out_dir}")

    ref = evaluate(LQIController())
    print("\n" + ENTETE)
    print(ligne("LQI (réf.)", ref))

    # 16 drones en parallèle, chacun avec sa graine et AVEC bruit de mesure
    env = make_vec_env(DroneEnv, n_envs=N_ENVS, seed=a.seed, env_kwargs=BRUIT_ENTRAINEMENT, vec_env_cls=DummyVecEnv)
    agent = PPO("MlpPolicy", env, seed=a.seed, device="cpu", verbose=0, tensorboard_log=str(out_dir), **PPO_KWARGS)
    # Politique initiale centrée sur le vol stationnaire : sortie moyenne u = U_EQ au lieu de 0 (chute libre).
    # Sans cela, l'agent débutant tombe et ne voit que des écarts énormes, où la récompense n'apprend presque rien.
    with torch.no_grad():
        agent.policy.action_net.bias.fill_(U_EQ)
    callback = EvalCallback(a.eval_freq, out_dir, ref)
    t0 = time.time()
    agent.learn(total_timesteps=a.steps, callback=callback, tb_log_name="tensorboard")
    duree = time.time() - t0

    best = PPO.load(out_dir / "best_model", device="cpu")
    final = evaluate(AgentController(best))
    print(f"\nEntraînement : {a.steps} transitions en {duree / 60:.1f} min. Meilleur agent (rendement {callback.best:.1f}) :")
    print(ENTETE)
    print(ligne("agent PPO", final))
    print(ligne("LQI (réf.)", ref))

    cwd = os.getcwd()
    os.chdir(out_dir)                                           # export_for_matlab écrit dans le dossier courant
    try:
        export_for_matlab(best, DroneEnv(z_ref_fixe=Z_EXPORT))  # SANS bruit, consigne fixe (reproduite par mission())
    finally:
        os.chdir(cwd)

    figures(callback.history, ref, AgentController(best), out_dir, nom_agent="agent PPO")
    json.dump({"reglages": dict(vars(a), n_envs=N_ENVS, ppo=PPO_KWARGS, bruit=BRUIT_ENTRAINEMENT, z_export=Z_EXPORT),
               "duree_min": duree / 60, "historique": callback.history, "agent_final": final, "lqi": ref},
              open(out_dir / "results.json", "w"), indent=2, default=str)
    print(f"\nÉcrit dans {out_dir} : best_model.zip, policy.mat, traj_model.mat, results.json, 2 figures")
    plt.show()


if __name__ == "__main__":
    main()
