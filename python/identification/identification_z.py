"""
Identification du drone par réseau récurrent (GRU) : commande u -> accélération verticale az.

    python identification_z.py                       # réglages par défaut (CONFIG ci-dessous)
    python identification_z.py --epochs 3 --seeds 0  # essai rapide

Principe
  - Le réseau ne reçoit QUE la commande u, jamais la vraie az : il est donc entraîné et évalué en
    simulation libre sur des essais entiers de 100 s (c'est exactement ce que l'agent RL lui fera subir).
  - Causalité : u(k) agit sur az(k+1). L'entrée du réseau à l'instant k est donc u(k-1), et sa sortie az(k).
  - Plusieurs graines ; la meilleure est choisie sur la VALIDATION, le TEST ne sert qu'au bilan final.

Sorties (dossier modeles/z/ à côté de ce fichier)
  - gru_z.pt      : poids PyTorch + configuration + normalisation
  - gru_z.npz     : les mêmes poids en numpy (pour l'environnement RL, sans PyTorch)
  - training.png, test_sequence.png, static_and_steps.png, z_drift.png
  - results.json      : tous les chiffres

Utilisation du modèle dans l'environnement RL (numpy pur, sans GPU) :
    from identification_z import DroneGRU
    model = DroneGRU.load("modeles/z/gru_z.npz")
    az = model.reset()        # az(0) au repos (u = 0 depuis longtemps)
    az = model.step(u)        # u(k) appliqué -> az(k+1)
"""
import sys
import argparse
import json
import math
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # dossier python/ (chemins.py)
from chemins import PYTHON  # noqa: E402  (rend importables les sous-dossiers de python/)
HERE = PYTHON                # racine python/ : modeles/, resultats/ ; HERE.parent = EI_RL/

DATA = HERE.parent / "donnees" / "nettoyees"
OUT = HERE / "modeles" / "z"
TEAMMATE_RNN = HERE.parent / "depot_groupe" / "identification_RNN" / "model_rnn" / "lightning_logs" / "version_3" / "export_NN"

TS = 0.05
G = 9.81
NOISE_FLOOR = 0.044          # écart-type du bruit sur az (étape 1) : plancher de la RMSE face aux mesures
Y_OFFSET, Y_SCALE = -G, 20.0  # az_n = (az - Y_OFFSET) / Y_SCALE : az de l'ordre de [-1, 1]

CONFIG = dict(
    cell="GRU",          # "GRU" ou "LSTM" (l'export numpy ne gère que GRU)
    hidden=64,           # neurones de la couche récurrente
    layers=1,            # couches récurrentes empilées
    epochs=150,          # époques maximum
    batch=32,            # essais (de 2000 pas) par lot
    lr=3e-3,             # pas d'apprentissage initial
    weight_decay=1e-6,
    clip=1.0,            # écrêtage de la norme du gradient (stabilise les RNN)
    patience=25,         # arrêt si la validation ne s'améliore plus pendant 25 époques
    seeds=[0, 1, 2],     # une graine = une initialisation et un ordre des lots différents
    split=[0.70, 0.15, 0.15],
    split_seed=0,        # mélange des essais AVANT découpage
)


# =============================================================================================
# 1. Données
# =============================================================================================
def load_data():
    """U, Y : (n_essais, 2000). pandas lit le .gz bien plus vite que np.loadtxt."""
    try:
        import pandas as pd
        U = pd.read_csv(DATA / "bdd_in_clean.csv.gz", header=None).to_numpy(np.float32)
        Y = pd.read_csv(DATA / "bdd_out_clean.csv.gz", header=None).to_numpy(np.float32)
    except ImportError:
        U = np.loadtxt(DATA / "bdd_in_clean.csv.gz", delimiter=",", dtype=np.float32)
        Y = np.loadtxt(DATA / "bdd_out_clean.csv.gz", delimiter=",", dtype=np.float32)
    assert U.shape == Y.shape
    return U, Y


def split_trials(n, fractions, seed):
    """Découpage PAR ESSAI (jamais des pas d'un même essai dans deux ensembles)."""
    idx = np.random.default_rng(seed).permutation(n)
    a, b = int(fractions[0] * n), int((fractions[0] + fractions[1]) * n)
    return idx[:a], idx[a:b], idx[b:]


def network_input(U):
    """Entrée à l'instant k = u(k-1) (u(-1) = 0 : repos). Forme (N, T, 1)."""
    X = np.zeros_like(U)
    X[:, 1:] = U[:, :-1]
    return X[..., None]


# =============================================================================================
# 2. Modèle
# =============================================================================================
class SeqModel(nn.Module):
    """Couche récurrente + tête linéaire : (N, T, 1) -> az normalisée (N, T)."""

    def __init__(self, cell, hidden, layers):
        super().__init__()
        rnn = {"GRU": nn.GRU, "LSTM": nn.LSTM}[cell]
        self.rnn = rnn(input_size=1, hidden_size=hidden, num_layers=layers, batch_first=True)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):
        h, _ = self.rnn(x)                 # état initial nul = drone au repos
        return self.head(h).squeeze(-1)


@torch.no_grad()
def predict(model, X, device, chunk=256):
    """Prédiction en m/s², par paquets pour ne pas saturer la mémoire du GPU."""
    model.eval()
    out = [model(torch.from_numpy(X[i:i + chunk]).to(device)).cpu().numpy() for i in range(0, len(X), chunk)]
    return np.concatenate(out) * Y_SCALE + Y_OFFSET


# =============================================================================================
# 3. Entraînement
# =============================================================================================
def train_one(seed, cfg, X, Yn, tr, va, Y, device):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = SeqModel(cfg["cell"], cfg["hidden"], cfg["layers"]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=8)

    # Tout le jeu d'entraînement tient sur le GPU (~17 Mo) : pas de DataLoader, on indexe directement
    Xg, Yg = torch.from_numpy(X).to(device), torch.from_numpy(Yn).to(device)
    best, best_state, best_epoch, hist = math.inf, None, 0, []
    t0 = time.time()
    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        perm = rng.permutation(tr)
        losses = []
        for i in range(0, len(perm), cfg["batch"]):
            b = torch.from_numpy(perm[i:i + cfg["batch"]]).to(device)
            loss = nn.functional.mse_loss(model(Xg[b]), Yg[b])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg["clip"])
            opt.step()
            losses.append(loss.item())
        train_rmse = math.sqrt(np.mean(losses)) * Y_SCALE
        val_rmse = float(np.sqrt(np.mean((predict(model, X[va], device) - Y[va]) ** 2)))
        sched.step(val_rmse)
        hist.append((epoch, train_rmse, val_rmse))
        star = ""
        if val_rmse < best:
            best, best_epoch, star = val_rmse, epoch, "  *"
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        print(f"  graine {seed} | époque {epoch:3d} | train {train_rmse:.4f} | val {val_rmse:.4f} m/s² | "
              f"lr {opt.param_groups[0]['lr']:.1e} | {time.time() - t0:5.0f} s{star}", flush=True)
        if epoch - best_epoch >= cfg["patience"]:
            print(f"  arrêt : pas d'amélioration depuis {cfg['patience']} époques")
            break
    model.load_state_dict(best_state)
    return model, dict(seed=seed, best_val_rmse=best, best_epoch=best_epoch, epochs_run=epoch,
                       train_time_s=round(time.time() - t0, 1)), np.array(hist)


# =============================================================================================
# 4. Évaluation
# =============================================================================================
def z_drift(err, starts, H=200):
    """Erreur sur z après H pas (10 s) en intégrant deux fois l'erreur sur az, fenêtres débutant à `starts`."""
    out = []
    for s in starts:
        e = err[:, s:s + H]
        out.append(np.cumsum(np.cumsum(e, axis=1), axis=1)[:, -1] * TS ** 2)
    return float(np.sqrt(np.mean(np.concatenate(out) ** 2)))


def evaluate(Yh, Y):
    err = Yh - Y
    return dict(rmse=float(np.sqrt(np.mean(err ** 2))), bias=float(err.mean()),
                z_drift_start=z_drift(err, [0]),                       # 10 s depuis le repos
                z_drift_anywhere=z_drift(err, range(200, 1800, 200)))  # 10 s n'importe où dans l'essai


def static_and_steps(predict_fn):
    """Caractéristique statique (az atteinte à u constant) et réponses indicielles."""
    T = 300
    u_grid = np.linspace(-1, 1, 201)
    U = np.zeros((len(u_grid), T), np.float32); U[:, 40:] = u_grid[:, None]
    az_eq = predict_fn(U)[:, -1]
    amps = np.array([-1, -0.5, 0.2, 0.39, 0.6, 1.0], np.float32)
    Us = np.zeros((len(amps), 120), np.float32); Us[:, 40:] = amps[:, None]
    return u_grid, az_eq, amps, predict_fn(Us)


# =============================================================================================
# 5. Version numpy (pour l'environnement RL) + RNN du camarade pour comparaison
# =============================================================================================
def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


class DroneGRU:
    """Le GRU entraîné, pas à pas, en numpy. Mêmes équations que torch.nn.GRU."""

    def __init__(self, params):
        self.p = params
        self.layers = int(params["layers"])
        self.hidden = int(params["hidden"])

    @classmethod
    def load(cls, path):
        return cls(dict(np.load(path)))

    def _cell(self, x):
        inp = np.atleast_1d(np.asarray(x, dtype=np.float64))
        for l in range(self.layers):
            Wi, Wh = self.p[f"weight_ih_l{l}"], self.p[f"weight_hh_l{l}"]
            bi, bh = self.p[f"bias_ih_l{l}"], self.p[f"bias_hh_l{l}"]
            gi, gh = Wi @ inp + bi, Wh @ self.h[l] + bh
            H = self.hidden
            r = _sigmoid(gi[:H] + gh[:H])
            z = _sigmoid(gi[H:2 * H] + gh[H:2 * H])
            n = np.tanh(gi[2 * H:] + r * gh[2 * H:])
            self.h[l] = (1 - z) * n + z * self.h[l]
            inp = self.h[l]
        y = float((self.p["head_weight"] @ inp + self.p["head_bias"])[0])
        return y * float(self.p["y_scale"]) + float(self.p["y_offset"])

    def reset(self):
        """Drone au repos (u = 0 depuis longtemps). Renvoie az(0)."""
        self.h = np.zeros((self.layers, self.hidden))
        return self._cell(0.0)

    def step(self, u):
        """u(k) appliqué pendant le pas -> az(k+1)."""
        return self._cell(u)


def export_numpy(model, cfg, path):
    sd = {k: v.detach().cpu().double().numpy() for k, v in model.state_dict().items()}
    params = {k.replace("rnn.", "").replace("head.", "head_"): v for k, v in sd.items()}
    params.update(layers=cfg["layers"], hidden=cfg["hidden"], y_offset=Y_OFFSET, y_scale=Y_SCALE)
    np.savez(path, **params)


def teammate_rnn(U):
    """RNN version_3 du dépôt du groupe (entrée u(k), h0 = 0), pour comparaison. None s'il est absent."""
    if not TEAMMATE_RNN.exists():
        return None

    def rd(name):
        txt = (TEAMMATE_RNN / name).read_text().strip().strip("[]")
        return np.array([[float(x) for x in row.split()] for row in txt.split(";")])

    Wih, Whh = rd("weight_ih_l0.txt").reshape(-1), rd("weight_hh_l0.txt")
    bh, Wo, bo = rd("bias_h.txt").reshape(-1), rd("head_weight.txt").reshape(-1), rd("head_bias.txt")[0, 0]
    h = np.zeros((len(U), len(bh))); out = np.empty(U.shape)
    for k in range(U.shape[1]):
        h = np.tanh(np.outer(U[:, k], Wih) + h @ Whh.T + bh)
        out[:, k] = (h @ Wo + bo) * 20.0 - 9.81
    return out


# =============================================================================================
# 6. Programme principal
# =============================================================================================
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int)
    p.add_argument("--hidden", type=int)
    p.add_argument("--layers", type=int)
    p.add_argument("--seeds", type=int, nargs="+")
    args = p.parse_args()
    cfg = dict(CONFIG, **{k: v for k, v in vars(args).items() if v is not None})

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
    print(f"Calcul sur : {device}" + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else
                                       "  -> pas de GPU détecté, l'entraînement sera lent"))
    OUT.mkdir(exist_ok=True)

    t0 = time.time()
    U, Y = load_data()
    tr, va, te = split_trials(len(U), cfg["split"], cfg["split_seed"])
    print(f"{len(U)} essais x {U.shape[1]} pas chargés en {time.time() - t0:.0f} s -> "
          f"train {len(tr)}, validation {len(va)}, test {len(te)}")
    X = network_input(U)
    Yn = ((Y - Y_OFFSET) / Y_SCALE).astype(np.float32)

    # --- entraînements (une graine chacun), choix sur la validation
    runs = []
    for seed in cfg["seeds"]:
        print(f"\n=== Entraînement, graine {seed} ===")
        model, info, hist = train_one(seed, cfg, X, Yn, tr, va, Y, device)
        runs.append((model, info, hist))
    print("\nGraine | meilleure époque | RMSE validation")
    for _, info, _ in runs:
        print(f"{info['seed']:6d} | {info['best_epoch']:6d} / {info['epochs_run']:<5d} | {info['best_val_rmse']:.4f} m/s²")
    model, info, hist = min(runs, key=lambda r: r[1]["best_val_rmse"])
    print(f"Retenu : graine {info['seed']}")

    # --- bilan sur le test (jamais vu)
    res = {"ours": evaluate(predict(model, X[te], device), Y[te])}
    Y_mate = teammate_rnn(U[te])
    if Y_mate is not None:
        res["teammate_v3"] = evaluate(Y_mate, Y[te])
    print(f"\nTEST ({len(te)} essais de 100 s, simulation libre) — plancher du bruit : {NOISE_FLOOR} m/s²")
    print(f"{'modèle':<14}{'RMSE':>10}{'biais':>10}{'dérive z 10 s (repos)':>24}{'dérive z 10 s (partout)':>26}")
    for name, r in res.items():
        print(f"{name:<14}{r['rmse']:>8.4f}  {r['bias']:>+8.4f}  {r['z_drift_start']:>20.3f} m  {r['z_drift_anywhere']:>22.3f} m")

    # --- caractéristique statique et réponses indicielles
    def pred_u(Uarr):
        return predict(model, network_input(Uarr.astype(np.float32)), device)
    u_grid, az_eq, amps, steps = static_and_steps(pred_u)
    u_eq = float(u_grid[np.argmin(np.abs(az_eq))])
    print(f"Vol stationnaire : u_eq = {u_eq:.3f} ; az(u=+1) = {az_eq[-1]:.2f}, az(u=-1) = {az_eq[0]:.2f} m/s²")

    # --- export + vérification numpy == torch
    torch.save({"state_dict": model.state_dict(), "config": cfg, "y_offset": Y_OFFSET, "y_scale": Y_SCALE,
                "input": "u(k-1)", "info": info}, OUT / "gru_z.pt")
    if cfg["cell"] == "GRU":
        export_numpy(model, cfg, OUT / "gru_z.npz")
        sim = DroneGRU.load(OUT / "gru_z.npz")
        u_seq = U[te[0], :300]
        az_np = [sim.reset()] + [sim.step(u) for u in u_seq[:-1]]
        az_torch = pred_u(u_seq[None])[0]
        print(f"Vérification numpy / PyTorch : écart max {np.max(np.abs(np.array(az_np) - az_torch)):.2e} m/s²")
    json.dump({"config": cfg, "chosen": info, "all_runs": [r[1] for r in runs], "test": res,
               "u_eq": u_eq}, open(OUT / "results.json", "w"), indent=2)

    # --- figures
    t = np.arange(U.shape[1]) * TS
    fig, ax = plt.subplots(figsize=(8, 4))
    for _, inf, h in runs:
        ax.plot(h[:, 0], h[:, 2], label=f"validation, graine {inf['seed']}")
    ax.plot(hist[:, 0], hist[:, 1], "k--", lw=1, label="entraînement (retenu)")
    ax.axhline(NOISE_FLOOR, color="gray", ls=":", label="bruit de mesure")
    ax.set_yscale("log"); ax.set_xlabel("époque"); ax.set_ylabel("RMSE az [m/s²]"); ax.legend(); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(OUT / "training.png", dpi=120)

    i = te[0]
    Yh = predict(model, X[[i]], device)[0]
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(11, 6), sharex=True, height_ratios=[1, 2])
    a1.step(t[:400], U[i, :400], where="post"); a1.set_ylabel("u")
    a2.plot(t[:400], Y[i, :400], ".", ms=3, color="gray", label="mesure")
    a2.plot(t[:400], Yh[:400], label="GRU (simulation libre)")
    if Y_mate is not None:
        a2.plot(t[:400], Y_mate[0, :400], "--", label="RNN du groupe (v3)")
    a2.set_ylabel("az [m/s²]"); a2.set_xlabel("t [s]"); a2.legend(); a2.set_title(f"Essai de test {i} (20 premières s)")
    fig.tight_layout(); fig.savefig(OUT / "test_sequence.png", dpi=120)

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.5))
    a1.plot(u_grid, az_eq); a1.axhline(0, color="k", lw=.5); a1.axvline(u_eq, color="k", ls="--", lw=.8)
    a1.set_xlabel("u"); a1.set_ylabel("az stationnaire [m/s²]"); a1.set_title(f"Caractéristique statique (u_eq = {u_eq:.3f})")
    for a, s in zip(amps, steps):
        a2.plot(np.arange(120) * TS, s, label=f"u = {a:+.2f}")
    a2.set_xlabel("t [s]"); a2.set_ylabel("az [m/s²]"); a2.set_title("Réponses indicielles (saut à t = 2 s)"); a2.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(OUT / "static_and_steps.png", dpi=120)

    err = predict(model, X[te], device) - Y[te]
    zerr = np.cumsum(np.cumsum(err[:, :200], axis=1), axis=1) * TS ** 2
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(t[:200], np.sqrt(np.mean(zerr ** 2, axis=0)), label="GRU")
    if Y_mate is not None:
        zm = np.cumsum(np.cumsum(Y_mate[:, :200] - Y[te, :200], axis=1), axis=1) * TS ** 2
        ax.plot(t[:200], np.sqrt(np.mean(zm ** 2, axis=0)), "--", label="RNN du groupe (v3)")
    ax.set_xlabel("t [s]"); ax.set_ylabel("RMSE de z [m]"); ax.set_title("Dérive de z en simulation libre (depuis le repos)")
    ax.legend(); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(OUT / "z_drift.png", dpi=120)
    print(f"\nÉcrit dans {OUT} : gru_z.pt, gru_z.npz, results.json, 4 figures. Durée totale {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
