"""Trois vignettes de l'animation de la mission (début de poursuite, descente, posé) pour la version PDF.
Lancer : python rapport/scripts/make_vignettes_gif.py   (lit rapport/animation_mission.gif, écrit rapport/figures/animation_vignettes.png)
"""
import sys
from pathlib import Path

from PIL import Image

sys.dont_write_bytecode = True
RAP = Path(__file__).resolve().parents[1]
g = Image.open(RAP / "animation_mission.gif")
n = g.n_frames
idx = [int(n * 0.15), int(n * 0.65), n - 1]
imgs = []
for i in idx:
    g.seek(i)
    imgs.append(g.convert("RGB").copy())
w = 700
imgs = [im.resize((w, int(im.height * w / im.width))) for im in imgs]
H = max(im.height for im in imgs)
sheet = Image.new("RGB", (w * 3 + 20, H), "white")
for k, im in enumerate(imgs):
    sheet.paste(im, (k * (w + 10), 0))
sheet.save(RAP / "figures" / "animation_vignettes.png")
print(n, idx, sheet.size)
