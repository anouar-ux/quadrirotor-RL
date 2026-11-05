%% Étape C2.4 : agents z + lacet + x sur UAV_model (4 moteurs), LQ d'attitude en boucle interne
%  Mission (40 s) : décollage à 0,5 m (1 s), cap tenu à 0, x : +2 m à 4 s, retour à 12 s, puis « voiture » 5,5 m
%  devant à 20 s qui roule à 0,3 m/s. y n'est pas asservi (phi tenu à 0 par le LQ) : étape C3.
%  1. vérifie controleur_x.m + LQ contre Python (traj_x.mat : observations, actions de l'agent x, commande u_theta) ;
%  2. vole la mission sur EINT_RL_closed_loop_x.slx (construit par construire_boucle_x.m s'il manque), sans bruit et avec
%     g_noise = 0,1, et compare au vol prédit par Python ; écrit resultats_x.mat et vol_x_simulink.png.
clear; close all; clc;
ici = fileparts(mfilename('fullpath')); cd(ici); addpath(ici, fullfile(ici, '..', '..', 'Matlab files'));
verifier_x(ici);          % vérification pas à pas contre Python (verifier_x.m)
if ~isfile(fullfile(ici, 'EINT_RL_closed_loop_x.slx')), construire_boucle_x(); end
random_init;
Ts = 0.05; T_end = 40;
P = load(fullfile(ici, 'traj_x.mat'));
V = struct();
for g = [0 0.1]
    g_noise = g;
    clear rl_controller_x controleur_x
    out = sim('EINT_RL_closed_loop_x');
    V.(sprintf('g%03d', round(1000 * g))) = out.rl_log_x;   % t, 18 sorties, u1..u4, z_ref, psi_ref, x_ref, theta_ref, u_z, u_theta
end
save(fullfile(ici, 'resultats_x.mat'), 'V', 'P');

% ---- indicateurs
fprintf('\n%-10s %10s %12s %14s %12s %12s %10s %10s\n', 'g_noise', 'RMSE x', 'écart Python', 'écart voiture', ...
        'dépass. voit.', 'RMSE z', 'psi max', '|y| max');
for nom = fieldnames(V)'
    L = V.(nom{1}); t = L(:, 1); x = L(:, 2); y = L(:, 3); z = L(:, 4); psi = L(:, 10); x_ref = L(:, 26); z_ref = L(:, 24);
    xp = interp1(P.t + 0.05, P.x_apres, t, 'previous', 'extrap');          % vol prédit par Python au même instant
    i_voit = t >= 30;                                                     % suivi de la voiture une fois rattrapée
    i_rat = t >= 20 & t < 30;
    fprintf('%-10s %9.3f m %10.3f m %12.3f m %10.3f m %10.3f m %8.3f %8.3f m\n', nom{1}, sqrt(mean((x_ref - x).^2)), ...
        max(abs(x - xp)), max(abs(x_ref(i_voit) - x(i_voit))), max(x(i_rat) - x_ref(i_rat)), ...
        sqrt(mean((z_ref(t > 3) - z(t > 3)).^2)), max(abs(psi)), max(abs(y)));
end

% ---- figure
L0 = V.g000; L1 = V.g100; t = L0(:, 1);
xp = interp1(P.t + 0.05, P.x_apres, t, 'previous', 'extrap');
figure('Position', [50 50 1200 900], 'Name', 'Agents z + lacet + x sur le simulateur');
theme(gcf, 'light');              % figures en mode clair, même si Windows / MATLAB sont en mode sombre
subplot(5, 1, 1); plot(t, L0(:, 26), 'k:'); hold on; plot(t, xp, 'Color', [.6 .6 .6], 'LineWidth', 2.5);
plot(t, L0(:, 2), t, L1(:, 2), '--', 'LineWidth', 1.1); grid on; ylabel('x [m]');
legend('consigne', 'Python (prédit)', 'Simulink g = 0', 'Simulink g = 0,1', 'Location', 'northwest');
title('Agent PPO x (entraîné aux saturations) + LQ d''attitude, avec les agents z et lacet');
subplot(5, 1, 2); plot(t, L0(:, 26) - L0(:, 2), t, L1(:, 26) - L1(:, 2), '--'); grid on; ylabel('erreur x [m]');
subplot(5, 1, 3); plot(t, L0(:, 27), 'k:', t, L0(:, 9), t, L1(:, 9), '--'); grid on; ylabel('\theta [rad]');
legend('\theta_{ref}', 'g = 0', 'g = 0,1');
subplot(5, 1, 4); plot(t, L0(:, 24), 'k:', t, L0(:, 4), t, L1(:, 4), '--'); grid on; ylabel('z [m]');
subplot(5, 1, 5); plot(t, L0(:, 3), t, L1(:, 3), '--', t, L0(:, 10), t, L1(:, 10), '--'); grid on;
ylabel('y [m], \psi [rad]'); legend('y, g = 0', 'y, g = 0,1', '\psi, g = 0', '\psi, g = 0,1'); xlabel('t [s]');
exportgraphics(gcf, fullfile(ici, 'vol_x_simulink.png'), 'Resolution', 120);
