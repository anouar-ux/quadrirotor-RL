"""
L'AGENT APPREND : GIF de l'agent z (altitude) au fil de son entraînement PPO.

1. Entraînement (si les étapes ne sont pas déjà enregistrées) : MÊMES réglages que entrainement/train_ppo_z.py
   (16 drones en parallèle, PPO 64 x 64, bruit de mesure, politique initiale centrée sur u = 0,39, 2 000 000 de
   transitions, graine 0). On enregistre la politique à quelques étapes choisies et le rendement d'évaluation tous
   les 50 000 pas. L'agent RETENU (resultats/agent_z_ppo/) n'est PAS modifié : tout va dans
   resultats/agent_z_ppo_apprentissage/. Ce réentraînement ne redonne pas exactement l'agent retenu (aléa des
   calculs), mais il suit la même progression.
2. Animation : pour chaque étape, la mission de la Figure 1 (0 -> 0,5 m à 3 s, -> -0,25 m à 7 s) sur le GRU de z,
   tracée en accéléré (x 3,3), avec la courbe d'apprentissage et un drone qui monte et descend ; les étapes précédentes
   restent en gris clair.

    python python/presentation/agent_apprend.py              # entraîne (~2 à 10 min) si besoin, puis anime
    python python/presentation/agent_apprend.py --reentrainer
Sorties : rapport/presentation/09_agent_apprend.gif, 09_agent_apprend.png (toutes les étapes sur une figure).
"""
import argparse
import json
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import animation, patches  # noqa: E402

import commun as C  # noqa: E402
from chemins import RESULTATS  # noqa: E402
from env_z import MISSION_FIG1, TS, DroneEnv, mission  # noqa: E402

DOSSIER = RESULTATS / "agent_z_ppo_apprentissage"
# étapes resserrées là où l'agent apprend (entre 40 000 et 100 000 transitions)
ETAPES = (0, 8_000, 16_000, 24_000, 32_000, 40_000, 48_000, 56_000, 64_000, 72_000, 80_000, 88_000, 96_000,
          112_000, 128_000, 160_000, 240_000, 400_000, 800_000, 2_000_000)
PAS_TOTAL, EVAL_TOUS = 2_000_000, 8_000
N_VOL = 200                      # 10 s de mission


def entrainer():
    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.vec_env import DummyVecEnv
    from train_ppo_z import N_ENVS, PPO_KWARGS, U_EQ
    from train_sac_z import BRUIT_ENTRAINEMENT, AgentController, evaluate

    DOSSIER.mkdir(parents=True, exist_ok=True)

    class Etapes(BaseCallback):
        def __init__(self):
            super().__init__()
            self.a_sauver, self.eval_suivante, self.hist = [e for e in ETAPES if e > 0], EVAL_TOUS, []

        def _on_step(self):
            n = self.num_timesteps
            while self.a_sauver and n >= self.a_sauver[0]:
                self.model.save(DOSSIER / f"etape_{self.a_sauver.pop(0)}");
            if n >= self.eval_suivante:
                self.eval_suivante += EVAL_TOUS
                self.hist.append((n, evaluate(AgentController(self.model))["rendement"]))
                print(f"{n:>8} pas : rendement {self.hist[-1][1]:8.1f}", flush=True)
            return True

    env = make_vec_env(DroneEnv, n_envs=N_ENVS, seed=0, env_kwargs=BRUIT_ENTRAINEMENT, vec_env_cls=DummyVecEnv)
    agent = PPO("MlpPolicy", env, seed=0, device="cpu", verbose=0, **PPO_KWARGS)
    with torch.no_grad():
        agent.policy.action_net.bias.fill_(U_EQ)
    agent.save(DOSSIER / "etape_0")
    cb = Etapes()
    cb.hist.append((0, evaluate(AgentController(agent))["rendement"]))
    t0 = time.time()
    agent.learn(total_timesteps=PAS_TOTAL, callback=cb)
    agent.save(DOSSIER / f"etape_{PAS_TOTAL}")
    (DOSSIER / "historique.json").write_text(json.dumps(dict(historique=cb.hist, duree_min=(time.time() - t0) / 60)), encoding="utf-8")
    print(f"entraînement : {(time.time() - t0) / 60:.1f} min -> {DOSSIER}")


def voler(etape):
    from stable_baselines3 import PPO
    ag = PPO.load(DOSSIER / f"etape_{etape}", device="cpu")
    env = DroneEnv(z_ref_fixe=MISSION_FIG1)
    obs, info = env.reset(seed=0)
    Z = [env.z]
    for _ in range(N_VOL):
        a = float(np.asarray(ag.predict(obs, deterministic=True)[0]).reshape(-1)[0])
        obs, *_, info = env.step(np.array([a], np.float32))
        Z.append(float(np.clip(env.z, -3, 3)))
    return np.array(Z)


def animer():
    C.style_clair()
    H = json.loads((DOSSIER / "historique.json").read_text(encoding="utf-8"))["historique"]
    hp, hr = np.array([h[0] for h in H]) / 1e6, np.array([h[1] for h in H])
    t = np.arange(N_VOL + 1) * TS
    zr = np.array([mission(x, MISSION_FIG1) for x in t])
    vols = {e: voler(e) for e in ETAPES}
    for e in ETAPES:
        print(f"étape {e:>8} : erreur RMS sur la mission {np.sqrt(np.mean((vols[e] - zr) ** 2)) * 100:6.1f} cm")
    cmap = plt.get_cmap("viridis")
    coul = {e: cmap(0.9 * k / (len(ETAPES) - 1)) for k, e in enumerate(ETAPES)}

    # ---- figure fixe : toutes les étapes
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.6), gridspec_kw=dict(width_ratios=(1.5, 1)))
    ax[0].plot(t, zr, "k--", lw=1.2, label="consigne (Figure 1)")
    for e in ETAPES:
        ax[0].plot(t, vols[e], color=coul[e], lw=1.6, label=f"après {e / 1e3:.0f} k pas")
    ax[0].set_ylim(-0.9, 1.1); ax[0].set_xlabel("t [s]"); ax[0].set_ylabel("z [m]"); ax[0].grid(alpha=0.4)
    ax[0].legend(fontsize=7.5, ncol=2, loc="upper right"); ax[0].set_title("la même mission, à différents moments de l'entraînement")
    ax[1].plot(hp, hr, color="0.3", lw=1.5)
    for e in ETAPES:
        k = int(np.argmin(np.abs(hp - e / 1e6))); ax[1].plot(hp[k], hr[k], "o", color=coul[e], ms=8, mec="k")
    ax[1].set_xlabel("transitions vécues [millions]"); ax[1].set_ylabel("rendement d'évaluation"); ax[1].grid(alpha=0.4)
    ax[1].set_title("courbe d'apprentissage (PPO, 16 drones simulés)")
    fig.suptitle("L'agent z apprend à piloter l'altitude", fontsize=12); fig.tight_layout()
    fig.savefig(C.PRESENTATION / "09_agent_apprend.png", dpi=130); plt.close(fig)

    # ---- GIF : chaque étape tracée en accéléré (x 3,3), étapes précédentes en gris
    IMG_VOL, PAUSE = 30, 8                                          # 30 images par vol de 10 s (x 3,3), 8 images de pause
    fig = plt.figure(figsize=(11, 5.4))
    axz = fig.add_axes((0.07, 0.12, 0.52, 0.74)); axd = fig.add_axes((0.60, 0.12, 0.06, 0.74)); axr = fig.add_axes((0.74, 0.12, 0.24, 0.74))
    fig.suptitle("L'agent z apprend à piloter l'altitude (mission de la Figure 1, modèle GRU)", fontsize=12)
    axz.plot(t, zr, "k--", lw=1.2, label="consigne")
    axz.set_ylim(-0.9, 1.1); axz.set_xlim(0, t[-1]); axz.set_xlabel("t [s]"); axz.set_ylabel("z [m]"); axz.grid(alpha=0.4)
    fant = [axz.plot([], [], color="0.75", lw=1)[0] for _ in ETAPES]
    ligne, = axz.plot([], [], lw=2.2)
    titre = axz.text(0.02, 0.97, "", transform=axz.transAxes, va="top", fontsize=12, weight="bold",
                     bbox=dict(fc="w", ec="0.7", alpha=0.9))
    axd.set_ylim(-0.9, 1.1); axd.set_xlim(-1, 1); axd.axis("off")
    axd.axhline(-0.85, color="0.5", lw=1)
    corps = patches.FancyBboxPatch((-0.6, -0.03), 1.2, 0.06, boxstyle="round,pad=0.01", fc="tab:blue", ec="k")
    axd.add_patch(corps)
    cible, = axd.plot([-0.9, 0.9], [0, 0], "k--", lw=1)
    axr.plot(hp, hr, color="0.3", lw=1.5); pt, = axr.plot([], [], "o", ms=11, mec="k")
    axr.set_xlabel("transitions vécues [millions]"); axr.set_ylabel("rendement d'évaluation"); axr.grid(alpha=0.4)
    axr.set_title("courbe d'apprentissage", fontsize=10)
    plan = [(k, j) for k in range(len(ETAPES)) for j in range(IMG_VOL + PAUSE)]

    def image(n):
        k, j = plan[n]
        e = ETAPES[k]
        i = min(int(round(j / IMG_VOL * N_VOL)), N_VOL)
        for m, f in enumerate(fant):
            f.set_data((t, vols[ETAPES[m]]) if m < k else ([], []))
        ligne.set_data(t[:i + 1], vols[e][:i + 1]); ligne.set_color(coul[e])
        titre.set_text(f"après {e / 1e3:,.0f} k transitions".replace(",", " ") if e else "avant l'entraînement (politique au hasard)")
        z = float(np.clip(vols[e][i], -0.85, 1.05))
        corps.set_y(z - 0.03); corps.set_facecolor(coul[e]); cible.set_ydata([zr[i], zr[i]])
        q = int(np.argmin(np.abs(hp - e / 1e6))); pt.set_data([hp[q]], [hr[q]]); pt.set_color(coul[e]); pt.set_markeredgecolor("k")
        return []

    an = animation.FuncAnimation(fig, image, frames=len(plan), interval=100, blit=False)
    C.enregistrer_gif(an, C.PRESENTATION / "09_agent_apprend.gif", fps=10, dpi=72)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--reentrainer", action="store_true")
    a = p.parse_args()
    if a.reentrainer or not (DOSSIER / "historique.json").exists():
        entrainer()
    animer()


if __name__ == "__main__":
    sys.exit(main())
