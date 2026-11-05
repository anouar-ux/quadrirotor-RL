function y = rl_controller_xy(v)
%RL_CONTROLLER_XY  Appelé à chaque pas par EINT_RL_closed_loop_xy.slx (bloc Interpreted MATLAB Function, période Ts).
%   v = [t ; 18 sorties nommées de UAV_model].
%   y = [u1..u4 ; z_ref ; psi_ref ; x_ref ; y_ref ; theta_ref ; phi_ref ; u_z_agent ; u_theta ; u_phi]  (13 sorties).
%   - agents z et lacet, position (x, y) par l'agent x sur les deux axes du drone (controleur_xy.m) ;
%   - LQ d'attitude RAPIDE (lq_attitude.mat, même gain pour les deux axes) : theta suit theta_ref, phi suit phi_ref ;
%   - anticipation de l'inclinaison sur z : u_z + U_EQ (1 / (cos theta cos phi) - 1) ;
%   - mélangeur « + » de l'étape A.
%   global MODE_XY = 'rl' (défaut) ou 'pd' (référence). "clear rl_controller_xy controleur_xy" avant chaque vol.
global MODE_XY
persistent K az_prev wdz_prev u_z_prev a_psi_prev wdy_prev wdx_prev Uth Uph
if isempty(K)
    S = load(fullfile(fileparts(mfilename('fullpath')), 'lq_attitude.mat'));
    K = S.K_ATT(:)';
    az_prev = 0; wdz_prev = 0; u_z_prev = 0; a_psi_prev = 0;
    wdy_prev = 0; wdx_prev = 0; Uth = zeros(5, 1); Uph = zeros(5, 1);
end
if isempty(MODE_XY), mode = 'rl'; else, mode = MODE_XY; end
U_PSI_MAX = 0.03; THETA_MAX = 0.2; U_EQ = 0.39;
t = v(1); m = v(2:end);
x = m(1); y_ = m(2); z = m(3); vx = m(4); vy = m(5); vz = m(6); phi = m(7); theta = m(8); psi = m(9);
wx = m(10); wy = m(11); wz = m(12); az = m(15); wdx = m(16); wdy = m(17); wdz = m(18);

[u_z, a_psi, a_av, a_co, refs] = controleur_xy(t, z, vz, az_prev, u_z_prev, psi, wz, wdz_prev, a_psi_prev, ...
                                               x, y_, vx, vy, theta, phi, wx, wy, mode);
theta_ref = THETA_MAX * a_av;
phi_ref = -THETA_MAX * a_co;
u_th = -K * [theta - theta_ref; wy; wdy; wdy_prev; Uth];      % comme sim_xy.Attitude.commande
u_ph = -K * [phi - phi_ref; wx; wdx; wdx_prev; Uph];
u_th = min(max(u_th, -0.5), 0.5); u_ph = min(max(u_ph, -0.5), 0.5);
u_psi = U_PSI_MAX * a_psi;
u_zc = u_z + U_EQ * (1 / (cos(theta) * cos(phi)) - 1);        % anticipation de la perte de poussée verticale

u = min(max(u_zc + [u_psi - u_th; -u_psi + u_ph; u_psi + u_th; -u_psi - u_ph], -1), 1);
Uth = [(u(3) - u(1)) / 2; Uth(1:4)];          % commandes RÉELLEMENT appliquées (après saturation)
Uph = [(u(2) - u(4)) / 2; Uph(1:4)];
az_prev = az; wdz_prev = wdz; u_z_prev = u_z; a_psi_prev = a_psi;
wdy_prev = wdy; wdx_prev = wdx;
y = [u; refs(1:4); theta_ref; phi_ref; u_z; u_th; u_ph];
end
