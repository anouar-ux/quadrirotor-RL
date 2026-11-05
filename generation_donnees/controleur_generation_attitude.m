function u = controleur_generation_attitude(v)
% Contrôleur de génération de données (tangage + roulis) : appelé à chaque pas par le modèle de generation_donnees_attitude.m.
%   z : LQI (gains / 2, comme controleur_generation_psi.m) vers 0.5 m
%   ψ : petit PD vers 0 (le lacet ne doit pas dériver pendant l'essai)
%   θ, φ : u = EXCITATION(k) - K_ANG * angle - K_W * vitesse angulaire
%          (contrairement au lacet, l'attitude DOIT être tenue : un drone penché accélère sans limite en x ou y)
% v = [t ; 18 sorties nommées de UAV_model]. u = commandes des 4 moteurs, mélangeur « + » de l'étape A :
%   u1 = u_z + u_psi - u_theta,  u2 = u_z - u_psi + u_phi,  u3 = u_z + u_psi + u_theta,  u4 = u_z - u_psi - u_phi
% ATTENTION : la sortie theta de UAV_model est inutilisable. Son bloc de bruit a Cov = seed_theta (entier de 1 à
% 10000 tiré par random_init.m) au lieu de 0.0001 : bruit d'écart-type ~30 rad à g_noise = 0.1. On ESTIME theta en
% intégrant wy (trapèzes), dont le bruit est faible (~0.005 rad/s) : dérive ~0.005 rad en 20 s. Le drone part à plat.
global EXC_THETA EXC_PHI K_ANG K_W T_DEBUT
persistent integ theta_est wy_prev
if isempty(integ), integ = 0; theta_est = 0; wy_prev = 0; end
TS = 0.05; TAU = 0.35;
t = v(1); y = v(2:end);
z = y(3); vz = y(6); phi = y(7); psi = y(9); wx = y(10); wy = y(11); wz = y(12);
if t > 0, theta_est = theta_est + TS / 2 * (wy_prev + wy); end
wy_prev = wy;
theta = theta_est;
s = round(t / TS) * TS;
z_ref = 0.5 * (1 - (1 + s / TAU) * exp(-s / TAU));
e = z_ref - z;
integ = integ + e * TS;
u_z = 0.38 + 0.93 * e - 0.44 * vz + 0.395 * integ;
u_psi = -0.002 * psi - 0.006 * wz;
u_theta = -K_ANG * theta - K_W * wy;
u_phi = -K_ANG * phi - K_W * wx;
k = round((t - T_DEBUT) / TS) + 1;                % numéro du pas d'excitation (1 au début)
if k >= 1 && k <= numel(EXC_THETA)
    u_theta = u_theta + EXC_THETA(k);
    u_phi = u_phi + EXC_PHI(k);
end
u = min(max(u_z + [u_psi - u_theta; -u_psi + u_phi; u_psi + u_theta; -u_psi - u_phi], -1), 1);
end
