%% LQI d'attitude (étape C1) : synthèse sur le modèle ARX IDENTIFIÉ (python/identification/identification_attitude.py)
%  Le même gain sert pour theta et phi : leurs ARX sont identiques (gain 171.8 rad/s² par unité, couplage nul).
%
%  Modèle (pas k, Ts = 0.05 s) :  wd(k+1) = a1 wd(k) + a2 wd(k-1) + b1 u(k) + b2 u(k-1) + ... + b6 u(k-5)
%                                 w(k+1)  = w(k) + Ts/2 (wd(k) + wd(k+1))         (trapèzes)
%                                 ang(k+1) = ang(k) + Ts/2 (w(k) + w(k+1))
%                                 I(k+1)  = I(k) + Ts (ang(k) - ang_ref)            (action intégrale)
%  État : x = [ang, w, wd(k), wd(k-1), u(k-1), ..., u(k-5), I] (10 états), TOUS mesurés ou mémorisés :
%  ang = phi mesuré ou theta estimé (intégrale de wy), w = wx / wy, wd = w_dot_x / w_dot_y, u passés en mémoire.
%  Pas d'observateur : u(k) = -K x(k), avec ang remplacé par ang - ang_ref.
%  Sortie : gains_LQ_ARX_attitude.mat (K, modèle, réglage).
clear; clc;
here = fileparts(mfilename('fullpath')); cd(here);
R_ = jsondecode(fileread(fullfile(here, '..', '..', 'python', 'modeles', 'attitude', 'results.json')));
Ts = 0.05;
a = (R_.theta.a(:) + R_.phi.a(:)) / 2;  b = (R_.theta.b(:) + R_.phi.b(:)) / 2;   % moyenne des deux axes (quasi identiques)
[A, B] = modele_arx(a, b, Ts);
n = size(A, 1);
fprintf('ARX moyen : a = %s, b = %s, gain %.1f rad/s² par unité\n', mat2str(a', 4), mat2str(b', 3), sum(b) / (1 - sum(a)));
fprintf('pôles en boucle ouverte : %s\n', mat2str(eig(A(1:9, 1:9))', 3));

% Réglages candidats : Q pénalise l'angle, la vitesse angulaire et l'intégrale ; R la commande
% q_I = 0 : LQ sans intégrale (K_I = 0). E_INT : intégration conditionnelle (seulement si |ang - ref| < E_INT), Inf = toujours
reglages = [ ...      % q_ang  q_w   q_I    R     E_INT
              100     5     20     5e4    Inf;
              100     5     20     5e4    0.02;
              100     5     20     5e4    0.005;
              100     5     0      5e4    Inf;
              100     5     0      1e5    Inf;
              100     5     0      2e4    Inf;
              100     10    0      2e4    Inf];
fprintf('\n  q_ang  q_w   q_I      R  E_INT |   K_ang    K_w   K_I | montée 10-90 %%  dépassement  temps à 2 %%  |u| max (échelon 0.1 rad)\n');
for i = 1:size(reglages, 1)
    K = synthese(A, B, reglages(i, :));
    s = echelon(A, B, K, 0.1, 200, reglages(i, 5));
    fprintf('%6g %5g %5g %8g %5g | %7.4f %6.4f %6.4f | %8.2f s %12.1f %% %10.2f s %10.4f\n', reglages(i, :), K(1), K(2), K(end), ...
        s.montee, s.depassement, s.reponse, s.umax);
end

% Réglage RETENU : LQ (pas d'intégrale). L'identification ne montre AUCUN couple parasite (u = 0 <=> wd = 0, couplage
% nul) : l'intégrale n'a rien à compenser et ne fait que charger pendant la montée (25 % de dépassement en LQI classique,
% comme le LQI de lacet). Si un biais apparaissait (masse décentrée...), on prendrait la ligne « LQI conditionnel ».
RETENU = reglages(7, :);
K = synthese(A, B, RETENU);
fprintf('\nRETENU : q_ang %g, q_w %g, R %g\nK = %s\n', RETENU([1 2 4]), mat2str(K, 5));
fprintf('pôles en boucle fermée : %s\n', mat2str(abs(eig(A(1:9, 1:9) - B(1:9) * K(1:9)))', 3));
save(fullfile(here, 'gains_LQ_ARX_attitude.mat'), 'K', 'A', 'B', 'a', 'b', 'Ts', 'RETENU');

figure('Visible', 'off', 'Position', [100 100 900 600]);
theme(gcf, 'light');              % figures en mode clair, même si Windows / MATLAB sont en mode sombre
cas = {reglages(1, :), 'LQI classique'; reglages(2, :), 'LQI conditionnel (|e| < 0.02 rad)'; RETENU, 'LQ retenu'};
for i = 1:3
    s = echelon(A, B, synthese(A, B, cas{i, 1}), 0.1, 100, cas{i, 1}(5));
    subplot(2, 1, 1); plot(s.t, s.ang, 'LineWidth', 1.3); hold on;
    subplot(2, 1, 2); stairs(s.t, s.u, 'LineWidth', 1.1); hold on;
end
subplot(2, 1, 1); yline(0.1, 'k:'); ylabel('angle [rad]'); grid on; legend(cas(:, 2), 'Location', 'southeast');
title('Échelon de consigne de 0.1 rad sur le modèle ARX identifié');
subplot(2, 1, 2); ylabel('u (commande différentielle)'); xlabel('t [s]'); grid on;
exportgraphics(gcf, fullfile(here, 'echelons_LQ_ARX_attitude.png'), 'Resolution', 130);

function K = synthese(A, B, r)
if r(3) == 0                                   % LQ : modèle sans l'état intégral
    K = [dlqr(A(1:9, 1:9), B(1:9), diag([r(1), r(2), zeros(1, 7)]), r(4)), 0];
else
    K = dlqr(A, B, diag([r(1), r(2), zeros(1, 7), r(3)]), r(4));
end
end

function [A, B] = modele_arx(a, b, Ts)
% x = [ang, w, wd, wd_1, u_1..u_5, I] ; wd(k+1) = a1 wd + a2 wd_1 + b1 u + b2 u_1 + ... + b6 u_5
Awd = [0, 0, a(1), a(2), b(2:6)', 0];  Bwd = b(1);         % ligne de wd(k+1)
A = zeros(10); B = zeros(10, 1);
A(3, :) = Awd; B(3) = Bwd;
A(2, :) = [0, 1, Ts/2, 0, zeros(1, 6)] + Ts/2 * Awd; B(2) = Ts/2 * Bwd;
A(1, :) = [1, Ts/2, 0, 0, zeros(1, 6)] + Ts/2 * A(2, :); B(1) = Ts/2 * B(2);
A(4, 3) = 1;                                               % wd_1(k+1) = wd(k)
B(5) = 1; A(6, 5) = 1; A(7, 6) = 1; A(8, 7) = 1; A(9, 8) = 1;   % u_1(k+1) = u(k), u_2(k+1) = u_1(k), ...
A(10, :) = [Ts, zeros(1, 8), 1];                           % I(k+1) = I(k) + Ts * ang(k)
end

function s = echelon(A, B, K, ref, N, e_int)
% Réponse à un échelon de consigne ref à k = 0 (erreur d'angle = ang - ref), intégration conditionnelle |e| < e_int
x = zeros(size(A, 1), 1); ang = zeros(N, 1); u = zeros(N, 1);
for k = 1:N
    xe = x; xe(1) = x(1) - ref;
    u(k) = -K * xe; ang(k) = x(1);
    I = x(10); x = A * x + B * u(k);
    x(10) = I + (abs(xe(1)) < e_int) * 0.05 * xe(1);     % l'intégrale porte sur ang - ref
end
t = (0:N - 1)' * 0.05;
s.montee = t(find(ang >= 0.9 * ref, 1)) - t(find(ang >= 0.1 * ref, 1));
s.depassement = 100 * (max(ang) - ref) / ref;
j = find(abs(ang - ref) > 0.02 * abs(ref), 1, 'last'); s.reponse = Inf; if j < N, s.reponse = t(j + 1); end
s.umax = max(abs(u)); s.ang = ang; s.u = u; s.t = t;
end
