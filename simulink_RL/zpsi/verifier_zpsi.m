function verifier_zpsi(ici)
% VERIFIER_ZPSI  Vérification pas à pas du contrôleur MATLAB de l'étape « zpsi » contre le vol Python
% exporté (traj_*.mat) : mêmes consignes, observations, actions et commandes. Appelé par EINT_RL_eval_zpsi.m
% et par ../verifier_tout.m. Argument : ici = dossier simulink_RL/zpsi/.
Z = load(fullfile(ici, 'traj_z.mat')); P = load(fullfile(ici, 'traj_psi.mat'));
clear controleur_zpsi
for k = 1:numel(Z.t)
    [u_z, a_psi, z_ref, psi_ref, oz, op] = controleur_zpsi(Z.t(k), Z.z(k), Z.vz(k), Z.az_prev(k), Z.u_prev(k), ...
                                                               P.psi(k), P.wz(k), P.wdz_prev(k), P.a_prev(k));
    ou = sprintf('pas %d (t = %.2f s)', k - 1, Z.t(k));
    if abs(z_ref - Z.z_ref(k)) > 1e-9 || abs(psi_ref - P.psi_ref(k)) > 1e-9
        error('%s : consigne différente de Python (z_ref %g / %g, psi_ref %g / %g).', ou, z_ref, Z.z_ref(k), psi_ref, P.psi_ref(k));
    end
    [ez, iz] = max(abs(oz' - Z.obs(k, :)) ./ max(1, abs(Z.obs(k, :))));
    [ep, ip] = max(abs(op' - P.obs(k, :)) ./ max(1, abs(P.obs(k, :))));
    if ez > 1e-5, error('%s : composante %d de obs_z = %g (Python %g).', ou, iz, oz(iz), Z.obs(k, iz)); end
    if ep > 1e-5, error('%s : composante %d de obs_psi = %g (Python %g).', ou, ip, op(ip), P.obs(k, ip)); end
    if abs(u_z - Z.u(k)) > 1e-4 || abs(a_psi - P.a(k)) > 1e-4
        error('%s : commandes différentes (u_z %g / %g, a_psi %g / %g).', ou, u_z, Z.u(k), a_psi, P.a(k));
    end
end
clear controleur_zpsi
fprintf('Vérification OK : controleur_zpsi.m = Python sur %d pas (z et lacet).\n', numel(Z.t));
end
