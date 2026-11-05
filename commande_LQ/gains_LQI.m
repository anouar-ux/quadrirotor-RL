%% GAINS DES LQI « DOUBLE INTÉGRATEUR » (commande par modèle « double intégrateur »)
%  À quoi sert ce script : calculer, dans l'espace de travail, tous les gains utilisés par les modèles Simulink de
%  commande_LQ/ (test_LQI_z.slx, test_LQI_psi.slx, test_LQI_x.slx, test_LQI_y.slx, mission_LQ/mission_LQ.slx).
%  À lancer AVANT d'ouvrir ou de simuler un de ces modèles. Il ajoute aussi « Matlab files » au chemin (UAV_model).
%
%  Méthode (pour chaque axe) :
%    1. modèle continu double intégrateur  d²(y)/dt² = k * u   (k identifié : voir aussi python/identification/identification_gain_k.py)
%       - z     : k_poussee identifié par moindres carrés sur les données nettoyées (a_z + g = k * u) ;
%       - psi   : k_psi = 2,93 * 180 / 3,14 rad/s² par unité (mesuré sur Simulink) ;
%       - theta : k_theta = k_psi (même mesure) ; phi utilise les mêmes gains que theta (moteurs 2 et 4) ;
%    2. discrétisation ZOH à Ts = 0,05 s, ajout d'un état intégral (LQI) : x = [y ; dy/dt ; intégrale de l'erreur] ;
%    3. gains par dlqr : u = -K x, soit Kp (position), Kd (vitesse), Ki (intégrale).
%  Boucle externe de position (x, y) : PD réglé à la main, theta_ref = Kp_x * erreur - Kd_x * vitesse.
%
%  Remplace LQI.m, ident_psi.m, LQI_theta.m et la partie « gains » de atterissage.m (mêmes calculs, mêmes résultats ;
%).
clc; clear; close all;
ici = fileparts(mfilename('fullpath'));
addpath(ici, fullfile(ici, '..', 'Matlab files'));       % UAV_model.slx, random_init.m (fichiers fournis)
random_init;

%% 0. Paramètres
Ts = 0.05;                % période d'échantillonnage [s]
V_max = 8;
theta_max = 0.2;          % inclinaison maximale demandée par la boucle de position [rad]
tau = 1.2;                % constante de temps du filtre de consigne de z [s]

%% 1. Gain de poussée k (axe z), identifié sur les données nettoyées
[u_data, y_data] = lire_donnees(fullfile(ici, '..', 'donnees', 'nettoyees'));
[~, N_traj] = size(u_data);
a_z_global = []; u_global = [];
for idx = 1:N_traj                                   % (calcul d'origine, conservé à l'identique)
    a_z = y_data(:, idx);
    if abs(mean(a_z(1:10)) + 9.81) < 1               % compensation de la gravité
        a_z = a_z + 9.81;
    end
    a_z_global = [a_z_global; a_z]; %#ok<AGROW>
    u_global = [u_global; u_data(:, idx)]; %#ok<AGROW>
end
k_poussee = u_global \ a_z_global;
k_psi = 2.93 * 180 / 3.14;                          % rad/s² par unité de commande
k_theta = 2.93 * 180 / 3.14;
fprintf('Gains identifiés : k_poussee = %.4f, k_psi = %.4f, k_theta = %.4f\n', k_poussee, k_psi, k_theta);

%% 2. Synthèse LQI (même fonction pour les trois axes)
[K_aug, A_id, B_id] = lqi_double_integrateur(k_poussee, Ts, diag([80, 25, 25]), 1);
Kp = 0.8 * K_aug(1); Kd = 0.8 * K_aug(2); Ki = 0.8 * K_aug(3);          % z : gains réduits de 20 %
[K_aug_psi, A_psi, B_psi] = lqi_double_integrateur(k_psi, Ts, diag([120, 40, 30]), 10);
Kp_psi = K_aug_psi(1); Kd_psi = K_aug_psi(2); Ki_psi = K_aug_psi(3);
[K_aug_theta, A_theta, B_theta] = lqi_double_integrateur(k_theta, Ts, diag([150, 15, 100]), 1);
Kp_theta = K_aug_theta(1); Kd_theta = K_aug_theta(2); Ki_theta = K_aug_theta(3);
Kp_x = 0.23; Kd_x = 0.28;                            % PD de position (x et y)

fprintf('z     : Kp = %.4f, Kd = %.4f, Ki = %.4f   (u = -Kp e_z - Kd v_z - Ki int e_z)\n', Kp, Kd, Ki);
fprintf('psi   : Kp_psi = %.4f, Kd_psi = %.4f, Ki_psi = %.4f\n', Kp_psi, Kd_psi, Ki_psi);
fprintf('theta : Kp_theta = %.4f, Kd_theta = %.4f, Ki_theta = %.4f   (phi : mêmes gains, moteurs 2 et 4)\n', ...
    Kp_theta, Kd_theta, Ki_theta);
fprintf('x, y  : Kp_x = %.2f, Kd_x = %.2f\n', Kp_x, Kd_x);

function [K, A, B] = lqi_double_integrateur(k, Ts, Q, R)
% Gains LQI [Kp, Kd, Ki] du double intégrateur d²y/dt² = k u, discrétisé (ZOH) et augmenté d'un intégrateur.
sys = c2d(ss([0 1; 0 0], [0; k], eye(2), 0), Ts, 'zoh');
A = sys.A; B = sys.B;
K = dlqr([A, zeros(2, 1); [1 0] * Ts, 1], [B; 0], Q, R);
end

function [u, y] = lire_donnees(dossier)
% Lit bdd_in_clean.csv / bdd_out_clean.csv ; s'ils manquent, décompresse les .csv.gz (versions publiées sur GitHub).
f_in = fullfile(dossier, 'bdd_in_clean.csv'); f_out = fullfile(dossier, 'bdd_out_clean.csv');
if ~isfile(f_in) || ~isfile(f_out)
    gunzip(fullfile(dossier, 'bdd_*_clean.csv.gz'), dossier);
end
u = readmatrix(f_in); y = readmatrix(f_out);
end
