function y = rl_controller_x(v)
%RL_CONTROLLER_X  Appelé à chaque pas par EINT_RL_closed_loop_x.slx (bloc Interpreted MATLAB Function, période Ts).
%   v = [t ; 18 sorties nommées de UAV_model].
%   y = [u1 ; u2 ; u3 ; u4 ; z_ref ; psi_ref ; x_ref ; theta_ref ; u_z_agent ; u_theta].
%   - agents z, lacet et x (controleur_x.m) ; l'agent x donne theta_ref ;
%   - LQ d'attitude RAPIDE (lq_attitude.mat, même gain pour les deux axes) : theta suit theta_ref, phi est tenu à 0
%     (la position y n'est pas encore asservie : étape C3) ;
%   - anticipation de l'inclinaison sur z : u_z + U_EQ (1 / (cos theta cos phi) - 1) ;
%   - mélangeur « + » de l'étape A.
%   "clear rl_controller_x controleur_x" au début de chaque vol.
persistent K az_prev wdz_prev u_z_prev a_psi_prev a_x_prev wdy_prev wdx_prev Uth Uph
if isempty(K)
    S = load(fullfile(fileparts(mfilename('fullpath')), 'lq_attitude.mat'));
    K = S.K_ATT(:)';
    az_prev = 0; wdz_prev = 0; u_z_prev = 0; a_psi_prev = 0; a_x_prev = 0;
    wdy_prev = 0; wdx_prev = 0; Uth = zeros(5, 1); Uph = zeros(5, 1);
end
U_PSI_MAX = 0.03; THETA_MAX = 0.2; U_EQ = 0.39;
t = v(1); m = v(2:end);
x = m(1); z = m(3); vx = m(4); vz = m(6); phi = m(7); theta = m(8); psi = m(9);
wx = m(10); wy = m(11); wz = m(12); az = m(15); wdx = m(16); wdy = m(17); wdz = m(18);

[u_z, a_psi, a_x, refs] = controleur_x(t, z, vz, az_prev, u_z_prev, psi, wz, wdz_prev, a_psi_prev, ...
                                       x, vx, theta, wy, a_x_prev);
theta_ref = THETA_MAX * a_x;
u_th = -K * [theta - theta_ref; wy; wdy; wdy_prev; Uth];      % comme BoucleTangage.step de env_x.py
u_ph = -K * [phi - 0; wx; wdx; wdx_prev; Uph];
u_th = min(max(u_th, -0.5), 0.5); u_ph = min(max(u_ph, -0.5), 0.5);
u_psi = U_PSI_MAX * a_psi;
u_zc = u_z + U_EQ * (1 / (cos(theta) * cos(phi)) - 1);        % anticipation de la perte de poussée verticale

u = min(max(u_zc + [u_psi - u_th; -u_psi + u_ph; u_psi + u_th; -u_psi - u_ph], -1), 1);
Uth = [(u(3) - u(1)) / 2; Uth(1:4)];          % commandes RÉELLEMENT appliquées (après saturation)
Uph = [(u(2) - u(4)) / 2; Uph(1:4)];
az_prev = az; wdz_prev = wdz; u_z_prev = u_z; a_psi_prev = a_psi; a_x_prev = a_x;
wdy_prev = wdy; wdx_prev = wdx;
y = [u; refs(1); refs(2); refs(3); theta_ref; u_z; u_th];
end
