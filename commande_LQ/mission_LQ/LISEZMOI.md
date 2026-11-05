# Mission : atterrissage sur une voiture qui roule

Version dérivée du modèle complet d'atterrissage et des gains de ../gains_LQI.m.

Scénario :
1. décollage à 0,5 m ;
2. tour complet sur soi-même à vitesse constante (scan) ;
3. poursuite de la voiture, qui roule en x à vitesse constante ;
4. descente ;
5. posé sur la voiture.

## Lancer

Dans MATLAB :

```matlab
run('commande_LQ/mission_LQ/essai_mission_LQ.m')
```

Le script simule la mission pour g_noise = 0, 0,01 et 0,1. Il affiche ensuite l'instant du posé, l'écart drone − voiture et la vitesse relative, puis écrit `mission_LQ.png` et `resultats_mission_LQ.mat`.

Pour simuler à la main : lancer `parametres_mission_LQ.m`, puis ouvrir `mission_LQ.slx` et cliquer Run (durée 50 s).

## Fichiers

| Fichier | Rôle |
|---|---|
| `mission_LQ.slx` | copie du modèle complet  + bloc `mission` + anticipation de vitesse + bloc `rotation_psi` |
| `parametres_mission_LQ.m` | gains (`../gains_LQI.m`) + paramètres de la voiture `x_v0`, `v_v`, `h_v` |
| `bloc_mission_LQ.m` | code du bloc `mission`, pour lecture (le modèle contient sa propre copie) |
| `essai_mission_LQ.m` | simulation pour 3 niveaux de bruit, bilan chiffré et figure |

## Différences avec le premier modèle complet d'atterrissage

### 1. Bloc `mission` (MATLAB Function, discret, Ts = 0,05 s)

- **Entrées :** temps (Clock), x, vx, z, et les constantes `x_v0`, `v_v`, `h_v`.
- **Sorties :** `x_ref`, `vx_ref`, `y_ref`, `z_ref`, `psi_ref`, `phase`, `x_v`.

Les consignes `x_ref`, `y_ref`, `z_ref` et `psi_ref` alimentent les Goto qui étaient laissés vides. `vx_ref` part vers un nouveau Goto `vx_ref`.

C'est une machine à états, dont la phase est mémorisée dans des variables `persistent` :

| Phase | Déclenchement | Consignes |
|---|---|---|
| 1. Décollage | t = 0 | z_ref passe de 0 à 0,5 m en rampe sur 5 s |
| 2. Scan | t = 5 s | ψ_ref passe de 0 à 2π en 10 s (0,63 rad/s), puis **reste à 2π** |
| 3. Poursuite | t = 15 s | x_ref = x_v − d0(1+s)e^(−s), avec s = (t − t3)/3 s et d0 l'écart à t3 |
| 4. Descente | écart < 5 cm et < 5 cm/s pendant 1 s | z_ref descend à 0,1 m/s jusqu'à h_v, en suivant toujours la voiture |
| 5. Posé | z − h_v < 2 cm | maintien |

Ce profil d'approche a été choisi car il démarre exactement sur la position du drone, avec une vitesse relative nulle. Il converge vers la voiture sans dépassement, et l'accélération demandée reste à d0/T² ≈ 0,6 m/s², bien en dessous de la saturation θ_max (environ 2 m/s²). Un échelon de 5,5 m aurait saturé θ et donné environ 30 % de dépassement.

### 2. Anticipation de vitesse dans la boucle x

- Avant : θ_ref = Kp·(x_ref − x) − Kd·vx.
- Après : θ_ref = Kp·(x_ref − x) + Kd·(vx_ref − vx).

Pour cela, un nouveau bloc `Sum_vx` (vx_ref − vx) est placé avant `Gain1`, et le gain de `Gain1` passe de `-Kd_x` à `Kd_x`.

Pourquoi : le drone n'a quasiment pas de traînée (ax ≈ 9,9·θ), donc maintenir une vitesse ne demande aucun θ. Si le PD freine sur vx sans connaître la vitesse visée, seule une erreur de position peut compenser ce freinage. Il reste alors une erreur de traînage e = Kd·v/Kp ≈ 1,2·v, soit 36 cm derrière la voiture à 0,3 m/s.

### 3. Bloc `rotation_psi` (entre Sum6/Sum8 et les saturations de θ_ref et φ_ref)

Les boucles x et y calculent une commande dans le repère **monde**, alors que θ et φ inclinent le drone dans **son** repère. Le bloc applique une rotation d'angle ψ :

```
c_x = theta_w ; c_y = -phi_w
theta_ref =  cos ψ·c_x + sin ψ·c_y
phi_ref   = -(-sin ψ·c_x + cos ψ·c_y)
```

Pourquoi c'est indispensable : dès que ψ ≈ π, un θ positif fait reculer le drone en x monde, et la boucle x devient une **rétroaction positive**. Sans bruit, l'erreur reste nulle et rien ne se voit. Avec g_noise = 0,01, sans cette rotation, le drone dérivait de 10 m en x et de 16 m en y pendant le scan.

### 4. Durée de simulation

Elle passe de 10 s à 50 s.

## Résultats (v_v = 0,3 m/s, x_v0 = 1 m)

| g_noise | Posé à | Écart x drone − voiture | Écart y | Vitesse relative | Dérive pendant le scan |
|---|---|---|---|---|---|
| 0 | 40,95 s | −0,9 cm | 0,0 cm | 0,003 m/s | 0 |
| 0,01 | 40,95 s | −1,0 cm | 0,0 cm | 0,002 m/s | < 2 mm |
| 0,1 | 40,60 s | −1,3 cm | −0,9 cm | 0,009 m/s | < 2,5 cm |

Les graines de bruit sont tirées au hasard par `random_init` à chaque lancement, donc les chiffres varient légèrement d'un essai à l'autre. Lors d'un second essai à g = 0,1, le drone s'est posé à 41,25 s, avec un écart de −0,4 cm et une vitesse relative de 0,007 m/s.

## Essai écarté

J'ai testé l'estimation θ = ∫ωy et φ = ∫ωx (utilisée dans `lqi_attitude`) à la place des sorties θ et φ. Elle n'est pas nécessaire dans ce modèle à pas variable, et elle est un peu moins bonne : à g = 0,1, l'écart au posé passe à 3,6 cm, car l'intégrale du gyroscope dérive avec son bruit. Ce modèle garde donc les sorties θ et φ.

## Limites connues

- **ψ dépasse de 0,26 rad (15°)** à la fin du scan : le LQI ψ suit la rampe en retard, puis la dépasse quand elle s'arrête. Une rampe adoucie à la fin réglerait le problème.
- **z descend à −0,12 m pendant les 0,3 premières secondes** (retard des moteurs au démarrage).
- **Au posé, z descend environ 3 cm sous h_v.** On peut ralentir la descente sur les derniers centimètres.
- **La voiture est supposée connue**, avec une trajectoire rectiligne uniforme. Pour une trajectoire quelconque, il faudra remplacer x_v(t) par la vraie position, ajouter y_v(t), et donner vy_ref à la boucle y, qui n'a pas encore d'anticipation de vitesse.
