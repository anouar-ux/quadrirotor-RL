%% MISSION FINALE (configuration finale : cap vers la voiture puis dans son sens, biais de l'agent x corrigé) : décollage, scan 360° en 10 s, poursuite d'une voiture (virages, vitesse variable), posé lent
%  Tous les contrôleurs sont des agents PPO (z, lacet, x sur les deux axes du drone) + LQ d'attitude ; le superviseur
%  à 6 phases (rl_controller_mission.m) enchaîne les phases sur les MESURES.
%  1. vérifie rl_controller_mission.m contre Python (traj_mission_cap.mat, écrit par python/export/export_mission.py) :
%     les mêmes entrées v donnent les mêmes sorties y (commandes moteur, phase, consignes) à chaque pas ;
%  2. vole la mission sur EINT_RL_closed_loop_mission.slx (créé par construire_boucle_mission.m s'il manque), avec les agents ('rl')
%     et avec la référence PD sur x/y ('pd'), sans bruit et avec g_noise = 0,1 ;
%     écrit resultats_mission_cap.mat ; graphes_mission.m trace mission_simulink_cap.png.
clear; close all; clc;
global MODE_MISSION
ici = fileparts(mfilename('fullpath')); cd(ici); addpath(ici, fullfile(ici, '..', '..', 'Matlab files'));
verifier_mission(ici);          % vérification pas à pas contre Python (verifier_mission.m)
if ~isfile(fullfile(ici, 'EINT_RL_closed_loop_mission.slx')), construire_boucle_mission(); end
random_init;
Ts = 0.05; T_end = 60;
P = load(fullfile(ici, 'traj_mission_cap.mat')); VO = load(fullfile(ici, 'voiture.mat'));
R = struct();
for mode = {'rl', 'pd'}
    for g = [0 0.1]
        g_noise = g; MODE_MISSION = mode{1};
        clear rl_controller_mission
        out = sim('EINT_RL_closed_loop_mission');
        R.(sprintf('%s_g%03d', mode{1}, round(1000 * g))) = out.rl_log_mission;
    end
end
MODE_MISSION = 'rl';
save(fullfile(ici, 'resultats_mission_cap.mat'), 'R', 'P');

% ---- bilan (colonnes : 1 t | 2 x 3 y 4 z 5 vx 6 vy 7 vz ... 9 theta 10 psi | 24 phase 25 z_ref 29 x_v 30 y_v)
noms = {'stabilisation', 'montée', 'scan', 'poursuite', 'descente', 'posé'};
fprintf('\n%-8s %7s %7s %8s %8s %6s | %9s %10s %9s | %9s %8s %8s\n', 'vol', 'scan', 'pours.', 'descente', 'posé', ...
        'pause', 'écart pos', 'v relative', 'vz pose', 'écart desc', 'dérive', 'err z');
for nom = fieldnames(R)'
    L = R.(nom{1}); t = L(:, 1); ph = L(:, 24);
    ti = nan(1, 6); for p = 1:5, j = find(ph >= p, 1); if ~isempty(j), ti(p + 1) = t(j); end, end
    eh = hypot(L(:, 2) - L(:, 29), L(:, 3) - L(:, 30));
    j = find(ph >= 5, 1);
    if isempty(j), ep = nan; vr = nan; vzp = nan;
    else
        i = min(round(t(j) / Ts) + 1, numel(VO.t));
        ep = eh(j); vr = hypot(L(j, 5) - VO.vx(i), L(j, 6) - VO.vy(i)); vzp = L(j, 7);
    end
    d = ph == 4; pause_s = sum(d & [abs(diff(L(:, 25))) < 1e-9; true]) * Ts;
    fprintf('%-8s %6.2fs %6.2fs %7.2fs %7.2fs %5.2fs | %7.3f m %8.3f m/s %7.3f m/s | %7.3f m %6.3f m %6.3f m\n', ...
        nom{1}, ti(3), ti(4), ti(5), ti(6), pause_s, ep, vr, vzp, max(eh(d)), max(hypot(L(ph == 2, 2), L(ph == 2, 3))), ...
        max(abs(L(ph >= 2 & ph <= 3, 4) - 0.5)));
end
graphes_mission(R, P, VO, ici);     % écart à Python + figure (relançable seul sur resultats_mission_cap.mat)
