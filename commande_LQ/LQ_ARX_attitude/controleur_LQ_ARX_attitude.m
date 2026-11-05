function y = controleur_LQ_ARX_attitude(v)
% Contrôleur d'essai de l'étape C1 : z tenu à 0.5 m (LQI de génération), psi tenu à 0 (PD), et theta, phi suivent
% leurs consignes avec le LQ d'attitude de gains_LQ_ARX_attitude.m (même gain K pour les deux axes).
% v = [t ; 18 sorties nommées de UAV_model]. y = [u1..u4 ; theta_ref ; phi_ref ; theta_est].
% État de chaque axe : [ang - ref, w, wd, wd(k-1), u(k-1), ..., u(k-5)] ; theta est estimé par l'intégrale de wy
% (la sortie theta de UAV_model est noyée dans le bruit, voir generation_attitude/controleur_generation_attitude.m).
global K_ATT
persistent integ theta_est wy_prev wdy_prev wdx_prev Uth Uph
if isempty(integ)
    integ = 0; theta_est = 0; wy_prev = 0; wdy_prev = 0; wdx_prev = 0; Uth = zeros(5, 1); Uph = zeros(5, 1);
end
TS = 0.05; TAU = 0.35;
t = v(1); m = v(2:end);
z = m(3); vz = m(6); phi = m(7); psi = m(9); wx = m(10); wy = m(11); wz = m(12); wdx = m(16); wdy = m(17);
if t > 0, theta_est = theta_est + TS / 2 * (wy_prev + wy); end
wy_prev = wy;

s = round(t / TS) * TS;
z_ref = 0.5 * (1 - (1 + s / TAU) * exp(-s / TAU));
e = z_ref - z;
integ = integ + e * TS;
u_z = 0.38 + 0.93 * e - 0.44 * vz + 0.395 * integ;
u_psi = -0.002 * psi - 0.006 * wz;

[theta_ref, phi_ref] = consignes(s);
u_th = -K_ATT(1:9) * [theta_est - theta_ref; wy; wdy; wdy_prev; Uth];
u_ph = -K_ATT(1:9) * [phi - phi_ref; wx; wdx; wdx_prev; Uph];
wdy_prev = wdy; wdx_prev = wdx;

u = min(max(u_z + [u_psi - u_th; -u_psi + u_ph; u_psi + u_th; -u_psi - u_ph], -1), 1);
Uth = [(u(3) - u(1)) / 2; Uth(1:4)];          % commandes RÉELLEMENT appliquées (après saturation), comme à l'identification
Uph = [(u(2) - u(4)) / 2; Uph(1:4)];
y = [u; theta_ref; phi_ref; theta_est];
end

function [th, ph] = consignes(t)
% Échelons de consigne (rad) : theta 0 -> 0.1 (3 s) -> -0.1 (5 s) -> 0 (7 s) ; phi 0 -> 0.1 (4 s) -> 0 (6 s) -> -0.05 (8 s)
th = 0.1 * (t >= 3) - 0.2 * (t >= 5) + 0.1 * (t >= 7);
ph = 0.1 * (t >= 4) - 0.1 * (t >= 6) - 0.05 * (t >= 8);
end
