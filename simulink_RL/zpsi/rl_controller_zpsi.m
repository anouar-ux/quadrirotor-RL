function y = rl_controller_zpsi(v)
%RL_CONTROLLER_AVANCE  Appelé à chaque pas par EINT_RL_closed_loop_zpsi.slx (bloc Interpreted MATLAB Function, période Ts).
%   v = [t ; 18 sorties nommées de UAV_model] ; y = [u1 ; u2 ; u3 ; u4 ; z_ref ; psi_ref].
%   Garde les mesures et commandes du pas précédent (az_prev, wdz_prev, u_z_prev, a_psi_prev : 0 au premier pas,
%   comme les UnitDelay de EINT_RL_closed_loop.slx) et applique le mélangeur « + » mesuré à l'étape A.
%   "clear rl_controller_zpsi controleur_zpsi" au début de chaque vol.
persistent az_prev wdz_prev u_z_prev a_psi_prev
if isempty(az_prev), az_prev = 0; wdz_prev = 0; u_z_prev = 0; a_psi_prev = 0; end
U_PSI_MAX = 0.03;                        % u_psi = U_PSI_MAX * a_psi, comme dans env_psi.py
t = v(1); m = v(2:end);
[u_z, a_psi, z_ref, psi_ref] = controleur_zpsi(t, m(3), m(6), az_prev, u_z_prev, m(9), m(12), wdz_prev, a_psi_prev);
u_psi = U_PSI_MAX * a_psi;
u = min(max(u_z + u_psi * [1; -1; 1; -1], -1), 1);   % lacet (+,-,+,-) ; tangage et roulis à 0 pour l'instant
az_prev = m(15); wdz_prev = m(18); u_z_prev = u_z; a_psi_prev = a_psi;
y = [u; z_ref; psi_ref];
end
