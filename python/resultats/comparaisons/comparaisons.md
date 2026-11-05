# Comparaisons LQI double intégrateur / LQ sur modèle ARX / agents RL

Drone simulé : modèles identifiés (GRU de z et du lacet, ARX de l'attitude), contrôleurs à 20 Hz, bruit g_noise = 0,1 pour les indicateurs « bruit ». Produit par python/evaluation/comparaisons_LQI_ARX_RL.py.

LQI double intégrateur : même loi que gains_LQI.m, gains réajustés pour 20 Hz (réponse calée sur Simulink, voir reajustement_gains_LQI.png et python/modeles/gains_LQI_20Hz.json).

## Altitude z

| contrôleur | montée | dépassement | RMSE Fig. 1 (bruit) | RMSE PRBS (bruit) | bande passante |
|---|---|---|---|---|---|
| LQI double intégrateur (gains réajustés à 20 Hz) | 0.85 s | 24.9 % | 11.7 cm | 8.3 cm | 0.47 Hz |
| LQI sur modèle ARX | 0.65 s | 10.0 % | 5.9 cm | 5.6 cm | 1.25 Hz |
| agent PPO z | 0.85 s | 0.2 % | 3.1 cm | 2.7 cm | 2.34 Hz |

## Lacet

| contrôleur | erreur pendant le scan (bruit) | dépassement en fin de tour | RMSE PRBS (bruit) | bande passante (vitesse) |
|---|---|---|---|---|
| LQI double intégrateur (gains réajustés à 20 Hz) | 9.75° | 15.0° | 8.0° | 0.55 Hz |
| LQ sur modèle ARX | 0.65° | 0.0° | 2.5° | > 1.64 Hz |
| agent PPO lacet | 0.39° | 0.1° | 1.9° | > 1.64 Hz |

## Attitude θ (échelon 0,1 rad sur l'ARX, 20 Hz)

| contrôleur | montée | dépassement | temps de réponse 2 % | diverge |
|---|---|---|---|---|
| LQI double intégrateur (gains réajustés à 20 Hz) | 0.25 s | 32 % | — | non |
| LQ sur ARX, réglage C1 (lent) | 0.85 s | 0 % | 1.45 s | non |
| LQ sur ARX, réglage rapide (boucle interne) | 0.30 s | 3 % | 0.80 s | non |

## Position x (cascade LQI : PD + le LQI θ réajusté ; PD ARX et agent : LQ ARX rapide en boucle interne)

| contrôleur | dépassement 2 m | dépassement 5 m | dépassement voiture | retard sur la voiture | RMSE PRBS (bruit) | bande passante |
|---|---|---|---|---|---|---|
| cascade LQI : PD 0,23/0,28 + LQI θ réajusté | 0.0 % | 0.0 % | 0.0 cm | 36.4 cm | 26.4 cm | 0.31 Hz |
| PD réglé sur l'ARX : 0,3/0,45 + anticipation | 0.4 % | 0.2 % | 9.8 cm | 0.1 cm | 7.3 cm | 1.17 Hz |
| agent PPO x (entraîné aux saturations) | 0.1 % | 0.0 % | 1.7 cm | 0.2 cm | 6.0 cm | 1.48 Hz |

## Mission (voiture rectiligne 0,3 m/s)

| mission | bruit | posé à | écart au contact | vitesse relative |
|---|---|---|---|---|
| LQ (Simulink, mission_LQ.slx) | g=0 | 41.0 s | 0.9 cm | 0.3 cm/s |
| LQ (Simulink, mission_LQ.slx) | g=0.01 | 41.0 s | 0.8 cm | 0.1 cm/s |
| LQ (Simulink, mission_LQ.slx) | g=0.1 | 41.1 s | 0.9 cm | 0.9 cm/s |
| RL (agents, simulateur assemblé) | g=0 | 29.7 s | 0.1 cm | 0.1 cm/s |
| RL (agents, simulateur assemblé) | g=0,1 | 29.7 s | 0.2 cm | 1.0 cm/s |
