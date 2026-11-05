%% Étape C3 : position (x, y) avec l'agent x réutilisé sur les deux axes du drone, erreurs tournées par psi
%  Mission (45 s) : décollage à 0,5 m (1 s) ; scan 360° entre 5 et 13 s en tenant (0, 0) ; à 16 s diagonale vers
%  (2 ; 1,5) ; à 24 s la « voiture » apparaît en (5,5 ; 0) et roule selon x à 0,3 m/s.
%  1. vérifie controleur_xy.m + LQ contre Python (traj_xy.mat, écrit par python/export/export_xy.py) ;
%  2. vole la mission sur EINT_RL_closed_loop_xy.slx (construit par construire_boucle_xy.m s'il manque) avec l'agent ('rl') et
%     avec la référence PD ('pd'), sans bruit et avec g_noise = 0,1 ; écrit resultats_xy.mat et vol_xy_simulink.png.
clear; close all; clc;
global MODE_XY
ici = fileparts(mfilename('fullpath')); cd(ici); addpath(ici, fullfile(ici, '..', '..', 'Matlab files'));
verifier_xy(ici);          % vérification pas à pas contre Python (verifier_xy.m)
if ~isfile(fullfile(ici, 'EINT_RL_closed_loop_xy.slx')), construire_boucle_xy(); end
random_init;
Ts = 0.05; T_end = 45;
P = load(fullfile(ici, 'traj_xy.mat'));
V = struct();
for mode = {'rl', 'pd'}
    for g = [0 0.1]
        g_noise = g; MODE_XY = mode{1};
        clear rl_controller_xy controleur_xy
        out = sim('EINT_RL_closed_loop_xy');
        V.(sprintf('%s_g%03d', mode{1}, round(1000 * g))) = out.rl_log_xy;
    end
end
MODE_XY = 'rl';
save(fullfile(ici, 'resultats_xy.mat'), 'V', 'P');

% ---- indicateurs (colonnes : 1 t | 2 x 3 y 4 z 5 vx 6 vy ... 8 phi 9 theta 10 psi | 24 z_ref 25 psi_ref 26 x_ref 27 y_ref)
fprintf('\n%-10s %8s %11s %10s %11s %11s %11s %8s %10s\n', 'vol', 'RMSE', 'dérive scan', 'dép. diag', 'dép. voit.', ...
        'écart voit.', 'écart Py', 'RMSE z', 'err psi');
for nom = fieldnames(V)'
    L = V.(nom{1}); t = L(:, 1);
    e = hypot(L(:, 26) - L(:, 2), L(:, 27) - L(:, 3));
    i_scan = t >= 5 & t < 14; i_diag = t >= 16 & t < 24; i_rat = t >= 24 & t < 35; i_voit = t >= 35;
    xp = interp1(P.t, P.x_vrai, t, 'previous', 'extrap'); yp = interp1(P.t, P.y_vrai, t, 'previous', 'extrap');
    fprintf('%-10s %6.3f m %9.3f m %8.3f m %9.3f m %9.3f m %9.3f m %6.3f m %8.3f\n', nom{1}, sqrt(mean(e(t >= 3).^2)), ...
        max(e(i_scan)), max([max(L(i_diag, 2)) - 2, max(L(i_diag, 3)) - 1.5, 0]), max([max(L(i_rat, 2) - L(i_rat, 26)), 0]), ...
        max(e(i_voit)), max(hypot(L(:, 2) - xp, L(:, 3) - yp)), sqrt(mean((L(t > 3, 24) - L(t > 3, 4)).^2)), ...
        max(abs(L(:, 25) - L(:, 10))));
end

% ---- figure
R0 = V.rl_g000; R1 = V.rl_g100; D1 = V.pd_g100; t = R0(:, 1);
figure('Position', [50 50 1250 900], 'Name', 'Étape C3 : position (x, y) sur le simulateur');
theme(gcf, 'light');              % figures en mode clair, même si Windows / MATLAB sont en mode sombre
subplot(3, 2, [1 3]); plot(R0(:, 26), R0(:, 27), 'k:', 'LineWidth', 1.2); hold on;
plot(P.x_vrai, P.y_vrai, 'Color', [.6 .6 .6], 'LineWidth', 2.5);
plot(R0(:, 2), R0(:, 3), R1(:, 2), R1(:, 3), '--', D1(:, 2), D1(:, 3), '-.', 'LineWidth', 1.1);
axis equal; grid on; xlabel('x [m]'); ylabel('y [m]'); title('Vue de dessus');
legend('consigne', 'Python (prédit)', 'agent, g = 0', 'agent, g = 0,1', 'PD, g = 0,1', 'Location', 'northeast');
subplot(3, 2, 2); plot(t, R0(:, 26), 'k:', t, R0(:, 2), t, R1(:, 2), '--', t, D1(:, 2), '-.'); grid on; ylabel('x [m]');
title('Agents z + lacet + x (sur les 2 axes) + LQ d''attitude');
subplot(3, 2, 4); plot(t, R0(:, 27), 'k:', t, R0(:, 3), t, R1(:, 3), '--', t, D1(:, 3), '-.'); grid on; ylabel('y [m]');
legend('consigne', 'agent, g = 0', 'agent, g = 0,1', 'PD, g = 0,1', 'Location', 'northeast');
subplot(3, 2, 5); semilogy(t, max(hypot(R0(:, 26) - R0(:, 2), R0(:, 27) - R0(:, 3)), 1e-4), ...
    t, max(hypot(R1(:, 26) - R1(:, 2), R1(:, 27) - R1(:, 3)), 1e-4), '--', ...
    t, max(hypot(D1(:, 26) - D1(:, 2), D1(:, 27) - D1(:, 3)), 1e-4), '-.'); grid on;
xregion(5, 13, 'FaceColor', [.85 .85 .85]); ylim([1e-3 5]); ylabel('|erreur| [m]'); xlabel('t [s]');
title('Erreur de position (zone grise : scan 360°)');
subplot(3, 2, 6); yyaxis left; plot(t, R0(:, 25), 'k:', t, R1(:, 10), '-'); ylabel('\psi [rad]');
yyaxis right; plot(t, R1(:, 8), t, R1(:, 9)); ylabel('\phi, \theta [rad]'); grid on; xlabel('t [s]');
legend('\psi_{ref}', '\psi (g = 0,1)', '\phi', '\theta', 'Location', 'northeast');
exportgraphics(gcf, fullfile(ici, 'vol_xy_simulink.png'), 'Resolution', 120);
