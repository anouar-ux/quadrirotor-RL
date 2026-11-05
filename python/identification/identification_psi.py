"""
Identification du LACET par réseau récurrent (GRU) : commande différentielle u_psi -> accélération de lacet wdz.
Même démarche que identification_z.py (z), sur les données générées par generation_donnees/generation_donnees_psi.m.

    python identification_psi.py
    python identification_psi.py --epochs 5 --seeds 0   # essai rapide

Différences avec z
  - u_psi est minuscule (|u_psi| < 0.03, gain ~170 rad/s² par unité) : l'entrée du réseau est NORMALISÉE,
    x = u_psi / U_SCALE. Le facteur est enregistré avec les poids et appliqué par LacetGRU.step().
  - Pas de gravité : wdz est centré sur 0 (Y_OFFSET = 0).
  - Comparaison à un modèle LINÉAIRE (ARX identifié par moindres carrés) : si le GRU ne fait pas mieux,
    la dynamique est linéaire et le modèle simple suffit.

Sorties (dossier modeles/psi/) : gru_psi.pt, gru_psi.npz, results.json, figures.
Utilisation dans l'environnement RL :
    from identification_psi import LacetGRU
    m = LacetGRU.load("modeles/psi/gru_psi.npz")
    wdz = m.reset()          # 0 : drone immobile en lacet
    wdz = m.step(u_psi)      # u_psi(k) appliqué -> wdz(k+1)
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
from scipy.io import loadmat
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # dossier python/ (chemins.py)
from chemins import PYTHON  # noqa: E402  (rend importables les sous-dossiers de python/)
HERE = PYTHON                # racine python/ : modeles/, resultats/ ; HERE.parent = EI_RL/
from identification_z import DroneGRU

DATA = HERE.parent / "donnees" / "lacet" / "donnees_lacet.mat"
OUT = HERE / "modeles" / "psi"
TS = 0.05
U_SCALE = 0.03               # u_psi / U_SCALE dans [-1, 1] (amplitude max des données)
Y_OFFSET, Y_SCALE = 0.0, 5.0  # wdz_n = wdz / 5 : de l'ordre de [-1, 1]

CONFIG = dict(hidden=32, layers=1, epochs=300, batch=16, lr=3e-3, weight_decay=1e-6, clip=1.0, patience=30,
              seeds=[0, 1, 2], split=[0.70, 0.15, 0.15], split_seed=0)


# ---------------------------------------------------------------------------------------------
# Données
# ---------------------------------------------------------------------------------------------
def load_data():
    D = loadmat(DATA, squeeze_me=True)
    return {k: np.atleast_2d(D[k]).astype(np.float32) for k in ("u_psi", "wdz", "wz", "psi", "z", "u_z")}, D


def split_trials(n, fractions, seed):
    idx = np.random.default_rng(seed).permutation(n)
    a, b = int(fractions[0] * n), int((fractions[0] + fractions[1]) * n)
    return idx[:a], idx[a:b], idx[b:]


def network_input(U):
    """Entrée à l'instant k = u_psi(k-1) / U_SCALE (repos avant l'essai). Forme (N, T, 1)."""
    X = np.zeros_like(U)
    X[:, 1:] = U[:, :-1] / U_SCALE
    return X[..., None]


# ---------------------------------------------------------------------------------------------
# Modèles
# ---------------------------------------------------------------------------------------------
class SeqModel(nn.Module):
    def __init__(self, hidden, layers):
        super().__init__()
        self.rnn = nn.GRU(input_size=1, hidden_size=hidden, num_layers=layers, batch_first=True)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):
        h, _ = self.rnn(x)
        return self.head(h).squeeze(-1)


@torch.no_grad()
def predict(model, X, device):
    model.eval()
    return model(torch.from_numpy(X).to(device)).cpu().numpy() * Y_SCALE + Y_OFFSET


def arx_fit(U, Y, na=2, nb=6):
    """ARX : wdz(k) = sum a_i wdz(k-i) + sum b_j u(k-j), moindres carrés (régression sur les MESURES)."""
    rows, tgt = [], []
    for u, y in zip(U, Y):
        for k in range(max(na, nb), len(y)):
            rows.append(np.r_[y[k - na:k][::-1], u[k - nb:k][::-1]]); tgt.append(y[k])
    theta, *_ = np.linalg.lstsq(np.array(rows), np.array(tgt), rcond=None)
    return theta[:na], theta[na:]


def arx_simulate(a, b, U):
    """Simulation LIBRE de l'ARX (il ne voit que u, comme le GRU)."""
    na, nb = len(a), len(b)
    Y = np.zeros_like(U, dtype=np.float64)
    for k in range(U.shape[1]):
        acc = np.zeros(len(U))
        for i in range(1, na + 1):
            if k - i >= 0: acc += a[i - 1] * Y[:, k - i]
        for j in range(1, nb + 1):
            if k - j >= 0: acc += b[j - 1] * U[:, k - j]
        Y[:, k] = acc
    return Y


# ---------------------------------------------------------------------------------------------
# Entraînement
# ---------------------------------------------------------------------------------------------
def train_one(seed, cfg, X, Yn, tr, va, Y, device):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = SeqModel(cfg["hidden"], cfg["layers"]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=10)
    Xg, Yg = torch.from_numpy(X).to(device), torch.from_numpy(Yn).to(device)
    best, best_state, best_epoch, hist, t0 = math.inf, None, 0, [], time.time()
    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        perm = rng.permutation(tr)
        for i in range(0, len(perm), cfg["batch"]):
            b = torch.from_numpy(perm[i:i + cfg["batch"]]).to(device)
            loss = nn.functional.mse_loss(model(Xg[b]), Yg[b])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg["clip"])
            opt.step()
        val = float(np.sqrt(np.mean((predict(model, X[va], device) - Y[va]) ** 2)))
        sched.step(val)
        hist.append((epoch, val))
        if val < best:
            best, best_epoch = val, epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        if epoch % 20 == 0 or epoch == 1:
            print(f"  graine {seed} | époque {epoch:3d} | val {val:.4f} rad/s² | meilleure {best:.4f} (époque {best_epoch}) | "
                  f"{time.time() - t0:4.0f} s", flush=True)
        if epoch - best_epoch >= cfg["patience"]:
            break
    model.load_state_dict(best_state)
    return model, dict(seed=seed, best_val_rmse=best, best_epoch=best_epoch, epochs_run=epoch), np.array(hist)


# ---------------------------------------------------------------------------------------------
# Évaluation
# ---------------------------------------------------------------------------------------------
def derive(err, H=200):
    """Erreur sur wz et psi après H pas (10 s), en intégrant l'erreur sur wdz, fenêtres partout dans l'essai."""
    e = np.concatenate([err[:, s:s + H] for s in range(0, err.shape[1] - H + 1, H)])
    w = np.cumsum(e, axis=1) * TS
    p = np.cumsum(w, axis=1) * TS
    return float(np.sqrt(np.mean(w[:, -1] ** 2))), float(np.sqrt(np.mean(p[:, -1] ** 2)))


def bilan(Yh, Y):
    err = Yh - Y
    dw, dp = derive(err)
    return dict(rmse=float(np.sqrt(np.mean(err ** 2))), biais=float(err.mean()), derive_wz_10s=dw, derive_psi_10s=dp)


# ---------------------------------------------------------------------------------------------
# Version numpy pour l'environnement RL
# ---------------------------------------------------------------------------------------------
class LacetGRU(DroneGRU):
    """Le GRU de lacet pas à pas en numpy : mêmes équations que DroneGRU, entrée normalisée par u_scale."""

    def reset(self):
        self.h = np.zeros((self.layers, self.hidden))
        return self._cell(0.0)

    def step(self, u_psi):
        return self._cell(u_psi / float(self.p["u_scale"]))


def export_numpy(model, cfg, path):
    sd = {k: v.detach().cpu().double().numpy() for k, v in model.state_dict().items()}
    params = {k.replace("rnn.", "").replace("head.", "head_"): v for k, v in sd.items()}
    params.update(layers=cfg["layers"], hidden=cfg["hidden"], y_offset=Y_OFFSET, y_scale=Y_SCALE, u_scale=U_SCALE)
    np.savez(path, **params)


# ---------------------------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int)
    p.add_argument("--hidden", type=int)
    p.add_argument("--seeds", type=int, nargs="+")
    cfg = dict(CONFIG, **{k: v for k, v in vars(p.parse_args()).items() if v is not None})
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    OUT.mkdir(exist_ok=True)
    t0 = time.time()

    S, D = load_data()
    U, Y = S["u_psi"], S["wdz"]
    tr, va, te = split_trials(len(U), cfg["split"], cfg["split_seed"])
    print(f"{len(U)} essais x {U.shape[1]} pas ({U.shape[1] * TS:.0f} s) ; train {len(tr)}, validation {len(va)}, "
          f"test {len(te)} ; calcul sur {device}")
    # plancher du bruit de mesure : wdz est la dérivée de wz, l'écart à la dérivée trapézoïdale lissée l'estime
    bruit = float(np.std(np.diff(Y[te], 2, axis=1)) / np.sqrt(6))
    print(f"Écart-type de wdz : {Y.std():.3f} rad/s² ; variation haute fréquence (majorant du bruit) : {bruit:.4f}")

    X = network_input(U)
    Yn = ((Y - Y_OFFSET) / Y_SCALE).astype(np.float32)

    # --- modèle linéaire de référence
    a, b = arx_fit(U[tr], Y[tr])
    res = {"arx": bilan(arx_simulate(a, b, U[te]), Y[te])}
    print(f"ARX(2,6) : a = {np.round(a, 4)}, b = {np.round(b, 2)} ; gain statique {b.sum() / (1 - a.sum()):.1f} rad/s² par unité")

    # --- GRU
    runs = []
    for seed in cfg["seeds"]:
        print(f"\n=== GRU, graine {seed} ===")
        runs.append(train_one(seed, cfg, X, Yn, tr, va, Y, device))
    model, info, hist = min(runs, key=lambda r: r[1]["best_val_rmse"])
    res["gru"] = bilan(predict(model, X[te], device), Y[te])
    print(f"\nRetenu : graine {info['seed']} (validation {info['best_val_rmse']:.4f})")
    print(f"\nTEST ({len(te)} essais, simulation libre)")
    print(f"{'modèle':<8}{'RMSE wdz':>12}{'biais':>10}{'dérive wz 10 s':>17}{'dérive psi 10 s':>18}")
    for name, r in res.items():
        print(f"{name:<8}{r['rmse']:>10.4f}  {r['biais']:>+9.4f}  {r['derive_wz_10s']:>12.3f} rad/s  {r['derive_psi_10s']:>12.3f} rad")

    # --- gain statique et linéarité (réponses à des échelons de u_psi)
    amps = np.array([-0.03, -0.01, -0.003, 0.003, 0.01, 0.03], np.float32)
    Us = np.zeros((len(amps), 60), np.float32); Us[:, 10:20] = amps[:, None]   # impulsion de 0.5 s
    Ws = predict(model, network_input(Us), device)
    gains = np.cumsum(Ws, axis=1)[:, -1] * TS / (amps * 10 * TS)               # Δwz / (u * durée)
    print("Gain (variation de wz après une impulsion de 0.5 s) / (u * 0.5 s) :", " ".join(f"{g:.0f}" for g in gains),
          "rad/s² par unité")

    # --- export + vérification numpy == torch
    torch.save({"state_dict": model.state_dict(), "config": cfg, "u_scale": U_SCALE, "y_offset": Y_OFFSET,
                "y_scale": Y_SCALE, "input": "u_psi(k-1)/u_scale", "info": info}, OUT / "gru_psi.pt")
    export_numpy(model, cfg, OUT / "gru_psi.npz")
    sim = LacetGRU.load(OUT / "gru_psi.npz")
    u_seq = U[te[0], :300]
    w_np = np.array([sim.reset()] + [sim.step(u) for u in u_seq[:-1]])
    w_t = predict(model, network_input(u_seq[None]), device)[0]
    print(f"Vérification numpy / PyTorch : écart max {np.max(np.abs(w_np - w_t)):.2e} rad/s²")
    json.dump({"config": cfg, "chosen": info, "all_runs": [r[1] for r in runs], "test": res, "arx": dict(a=a.tolist(), b=b.tolist()),
               "gains_impulsion": dict(zip([float(x) for x in amps], [float(g) for g in gains])), "bruit_hf": bruit},
              open(OUT / "results.json", "w"), indent=2)

    # --- figures : un essai de test (wdz, wz, psi reconstruits) et dérive
    i = te[0]; t = np.arange(U.shape[1]) * TS
    Wg = predict(model, X[[i]], device)[0]; Wa = arx_simulate(a, b, U[[i]])[0]
    integ = lambda w: np.cumsum(np.r_[0, (w[1:] + w[:-1]) / 2]) * TS        # trapèzes, comme l'environnement
    fig, ax = plt.subplots(4, 1, figsize=(11, 9), sharex=True)
    ax[0].step(t, U[i], where="post", lw=.8); ax[0].set_ylabel("u_psi")
    ax[1].plot(t, Y[i], ".", ms=2, color="gray", label="mesure Simulink"); ax[1].plot(t, Wg, lw=1, label="GRU")
    ax[1].plot(t, Wa, "--", lw=1, label="ARX"); ax[1].set_ylabel("wdz [rad/s²]"); ax[1].legend(fontsize=8)
    ax[2].plot(t, S["wz"][i], color="gray", label="mesure"); ax[2].plot(t, integ(Wg), label="GRU intégré")
    ax[2].plot(t, integ(Wa), "--", label="ARX intégré"); ax[2].set_ylabel("wz [rad/s]"); ax[2].legend(fontsize=8)
    ax[3].plot(t, S["psi"][i] - S["psi"][i][0], color="gray", label="mesure")
    ax[3].plot(t, integ(integ(Wg)), label="GRU intégré deux fois"); ax[3].plot(t, integ(integ(Wa)), "--", label="ARX")
    ax[3].set_ylabel("psi [rad]"); ax[3].set_xlabel("t [s]"); ax[3].legend(fontsize=8)
    for a_ in ax: a_.grid(alpha=.3)
    ax[0].set_title(f"Essai de test {i} : simulation libre sur 60 s (le modèle ne voit que u_psi)")
    fig.tight_layout(); fig.savefig(OUT / "test_sequence.png", dpi=110)

    fig, ax = plt.subplots(figsize=(8, 4))
    for _, inf, h in runs:
        ax.plot(h[:, 0], h[:, 1], label=f"validation, graine {inf['seed']}")
    ax.axhline(res["arx"]["rmse"], color="k", ls="--", label="ARX (test)")
    ax.set_yscale("log"); ax.set_xlabel("époque"); ax.set_ylabel("RMSE wdz [rad/s²]"); ax.legend(); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(OUT / "training.png", dpi=110)
    print(f"\nÉcrit dans {OUT}. Durée {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
