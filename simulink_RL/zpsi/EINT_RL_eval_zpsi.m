%% Tâche avancée : agents z + lacet sur UAV_model (décollage, un tour complet de scan, atterrissage)
%  1. vérifie controleur_zpsi.m contre Python (traj_z.mat, traj_psi.mat : mêmes consignes, observations, commandes)
%  2. vole la mission sur EINT_RL_closed_loop_zpsi.slx (construit par construire_boucle_zpsi.m s'il manque) et trace.
clear; close all; clc;
ici = fileparts(mfilename('fullpath')); cd(ici); addpath(ici, fullfile(ici, '..', '..', 'Matlab files'));
verifier_zpsi(ici);          % vérification pas à pas contre Python (verifier_zpsi.m)
if ~isfile(fullfile(ici, 'EINT_RL_closed_loop_zpsi.slx')), construire_boucle_zpsi(); end
random_init;
Ts = 0.05; T_end = 20;
g_noise = 0.1;                    % bruit réaliste (étape A)

clear rl_controller_zpsi controleur_zpsi
out = sim('EINT_RL_closed_loop_zpsi');
L = out.rl_log_zpsi;            % t, 18 sorties, u1..u4, z_ref, psi_ref
t = L(:, 1); z = L(:, 4); psi = L(:, 10); U = L(:, 20:23); z_ref = L(:, 24); psi_ref = L(:, 25);
figure('Name', 'Agents z + lacet sur le simulateur');
theme(gcf, 'light');              % figures en mode clair, même si Windows / MATLAB sont en mode sombre
subplot(3, 1, 1); plot(t, z_ref, '--', t, z, 'LineWidth', 1.2); yline(-0.25, ':'); grid on; ylabel('z [m]');
title(sprintf('g\\_noise = %g : RMSE z %.3f m, RMSE \\psi %.3f rad', g_noise, ...
    sqrt(mean((z_ref - z).^2)), sqrt(mean((psi_ref - psi).^2))));
subplot(3, 1, 2); plot(t, psi_ref, '--', t, psi, 'LineWidth', 1.2); grid on; ylabel('\psi [rad]');
subplot(3, 1, 3); stairs(t, U); grid on; ylabel('u_1 .. u_4'); xlabel('t [s]');
