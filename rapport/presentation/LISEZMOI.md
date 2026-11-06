# Présentation finale : animations et figures

Toutes ces images sont produites par les scripts de `python/presentation/`. Les vols utilisent le simulateur
assemblé à partir des modèles identifiés (GRU de z et du lacet, ARX de l'attitude), qui est le même que celui de
`sim_mission.py`. Les mesures sont bruitées comme avec `g_noise = 0,1` et le contrôleur tourne à 20 Hz. Les figures
sont en mode clair. Les GIF sont en temps réel, à 10 images/s, sauf mention contraire.

Les **trois familles** de contrôleurs comparées, sur tous les axes (`python/presentation/commun.py`) :

| couleur | famille | z | lacet ψ | position | attitude θ, φ |
|---|---|---|---|---|---|
| bleu | **agents RL (PPO)** | agent z | agent lacet | agent x sur les 2 axes du drone | LQ sur ARX |
| vert | **LQ sur modèle ARX** | LQI sur ARX | LQ sur ARX | PD 0,3 / 0,45 + vitesse de consigne | LQ sur ARX |
| rouge | **LQI « double intégrateur »** (gains réajustés à 20 Hz) | LQI | LQI | PD 0,23 / 0,28 + vitesse de consigne | LQI |

Le superviseur, le mélangeur et l'anticipation de poussée 1/cos sont communs aux trois familles.

## Diaporama de la soutenance

`presentation_finale.pptx` : la présentation finale en anglais, soit 18 slides (environ 11 min) et 3 slides de réserve (B1 à B3). Les notes d'orateur de chaque slide (minutage et chiffres à citer) sont dans les notes PowerPoint.

- **Modifiable** : chaque titre, texte, encadré et figure est un objet PowerPoint distinct. Les figures sont des images, qu'on remplace par clic droit puis « Changer l'image ».
- **Lisible partout** : le diaporama n'utilise que Arial et Georgia, présentes sous Windows, macOS et Google Slides. Il n'y a donc pas de substitution de police chez la personne qui l'ouvre.
- **Animations** : les 6 animations (slides 1, 8, 12, 13, 15 et 16) sont des GIF intégrés au fichier. Elles se lancent seules en mode diaporama dans PowerPoint, Keynote, Google Slides et LibreOffice, sans fichier externe. Elles sont compressées sans perte visible avec `gifsicle -O3 --lossy=35`, avec les mêmes images et la même cadence. Le fichier pèse ainsi 21 Mo, sous la limite de 25 Mo d'une pièce jointe Gmail.

## Fichiers

| fichier | ce qu'on y voit | message pour l'oral |
|---|---|---|
| `01_suivi_ligne_droite.gif` | Le drone (RL) reste au-dessus d'une voiture qui démarre sec, freine, recule, repart. Bandeaux « COUP DE FREIN ! », « DÉMARRAGE SEC ! » ; écart en cm. | Les à-coups atteignent 2 m/s², la limite physique du drone (θ_max = 0,2 rad). Il décroche de 20 à 30 cm pendant 0,5 s, puis revient. |
| `02_suivi_slalom.gif` | Même chose avec un slalom serré (rayon 1,2 m) puis un demi-tour (rayon 0,9 m). Le drone tourne avec la voiture. | Hors du premier démarrage, l'écart reste sous 10 cm. |
| `03_duel_ligne_droite.gif`, `04_duel_slalom.gif` | Les trois familles sur la même voiture et avec le même bruit, en trois vues côte à côte. Courbes d'écart superposées et RMS en direct. | Le RL est le plus précis dans les deux conduites. Le LQI « double intégrateur », conçu sans modèle des moteurs, oscille le plus. |
| `05_duel_bilan.png` | Écarts RMS et maximaux des duels. | Ligne droite, RMS : 12,1 (RL), 15,1 (ARX), 16,3 cm (LQI). Slalom : 5,4 / 6,3 / 8,9 cm. |
| `06_cible_atterrissage.png` | Mission complète sur 13 voitures (nominale et 12 aléatoires) pour les 3 familles. Points de posé sur la plateforme, avec un zoom, et instant du posé. | 39 posés sur 39, tous à moins de 5 cm du centre. Le RL se pose en 30 s en moyenne, contre 33 s (ARX) et 42,5 s (LQI). Le LQI est le plus précis (0,8 cm) mais le plus lent. |
| `06b_cible_atterrissage_rapide.png` (figure en anglais) | Même mission sur 100 voitures aléatoires (graines 1000 à 1099) qui changent de vitesse (0,5 à 1,0 m/s) et de virage (rayon ≥ 2,5 m) jusqu'à la fin du vol, pour les 3 familles : taux de réussite, points de posé, instant du posé. Le trajet aléatoire jusqu'au bout évite qu'un contrôleur lent se pose sur une voiture redevenue calme. | Posés en moins de 60 s : 100 / 100 / 78 % (RL / ARX / LQI) ; sur la plateforme : 100 / 97 / 73 % ; à moins de 5 cm : 96 / 73 / 49 %. Écart moyen 2,3 / 3,5 / 4,2 cm, posé en 33,5 / 37,3 / 49,2 s. Les 22 vols LQI non posés ne divergent pas : le drone reste à 10–15 cm de la voiture sans valider l'alignement (cap > 3°) ou la descente (pause au-delà de 10 cm). |
| `07_sous_le_capot.gif` | Démarrage, frein et marche arrière (RL) : silhouettes vues de côté et de derrière (inclinaison ×2), poussée des 4 moteurs, θ demandé par l'agent et θ obtenu. | Pour accélérer, le drone se penche. Le moteur arrière pousse plus pour basculer, puis le LQ contre-braque avant d'arriver à l'angle demandé, car les moteurs ont environ 3 pas de retard. |
| `08_architecture.png` (`.svg`) | Schéma-blocs : 6 phases du superviseur et leurs conditions, puis la chaîne agents → LQ d'attitude → mélangeur → moteurs. | Ce qui est appris (bleu), ce qui est modélisé (vert), ce qui est logique ou géométrie. |
| `09_agent_apprend.gif`, `09_agent_apprend.png` | La mission de la Figure 1 volée par l'agent z à 20 moments de son entraînement (accéléré ×3,3), avec la courbe d'apprentissage (évaluée toutes les 8 000 transitions). | Jusqu'à 40 000 transitions, le drone tombe ou s'envole. Tout se joue entre 48 000 et 100 000 : l'erreur RMS sur la mission passe de 161 cm à 16 cm (64 000), puis 7 cm (96 000). À 400 000 elle vaut 3 cm ; la suite (jusqu'à 2 millions) ne fait qu'affiner. |
| `suivi_voiture.json`, `cible_atterrissage.json`, `cible_atterrissage_rapide.json` | Indicateurs chiffrés. | |

## Refaire les images

```bash
python python/presentation/suivi_voiture.py          # 01 à 05 (≈ 8 min ; --rapide : une image sur 4)
python python/presentation/cible_atterrissage.py     # 06 (≈ 3 min ; --figure : redessine depuis le .json)
python python/presentation/cible_atterrissage_rapide.py   # 06b (≈ 2 min ; --figure : redessine depuis le .json)
python python/presentation/sous_le_capot.py          # 07 (1 min)
python python/presentation/architecture.py           # 08
python python/presentation/agent_apprend.py          # 09 (réentraîne z la 1re fois, dans resultats/agent_z_ppo_apprentissage/)
```

L'animation de la mission complète sur Simulink, avec poursuite, cap et posé, reste `python/resultats/mission/animation_mission.gif`. Elle
est produite par `python/evaluation/animation_mission.py`.
