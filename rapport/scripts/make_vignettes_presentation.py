"""Une vignette par GIF de rapport/presentation/ (version PDF). Lancer : python rapport/scripts/make_vignettes_presentation.py"""
import sys
from pathlib import Path

from PIL import Image

sys.dont_write_bytecode = True
RAP = Path(__file__).resolve().parents[1]
CHOIX = {"01_suivi_ligne_droite": 0.45, "02_suivi_slalom": 0.55, "03_duel_ligne_droite": 0.6, "04_duel_slalom": 0.6,
         "07_sous_le_capot": 0.4, "09_agent_apprend": 0.95}
for nom, f in CHOIX.items():
    g = Image.open(RAP / "presentation" / f"{nom}.gif")
    g.seek(int((g.n_frames - 1) * f))
    im = g.convert("RGB")
    w = 1100
    im.resize((w, int(im.height * w / im.width))).save(RAP / "figures" / f"vignette_{nom}.png")
    print(nom, g.n_frames, im.size)
