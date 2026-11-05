function verifier_x(ici)
% VERIFIER_X  Vérification pas à pas du contrôleur MATLAB de l'étape « x » contre le vol Python
% exporté (traj_*.mat) : mêmes consignes, observations, actions et commandes. Appelé par EINT_RL_eval_x.m
% et par ../verifier_tout.m. Argument : ici = dossier simulink_RL/x/.
% Rejoue le vol Python (traj_x.mat) dans controleur_x.m : consigne, observation et action de l'agent x, puis la
% commande u_theta du LQ d'attitude, pas par pas. (z et lacet sont vérifiés par simulink_RL/zpsi/EINT_RL_eval_zpsi.m.)
P = load(fullfile(ici, 'traj_x.mat')); S = load(fullfile(ici, 'lq_attitude.mat')); K = S.K_ATT(:)';
clear controleur_x
for k = 1:numel(P.t)
    [~, ~, a_x, refs, ox] = controleur_x(P.t(k), 0.5, 0, 0, 0.39, 0, 0, 0, 0, P.x(k), P.vx(k), P.theta(k), P.wy(k), P.a_prev(k));
    ou = sprintf('pas %d (t = %.2f s)', k - 1, P.t(k));
    if abs(refs(3) - P.x_ref(k)) > 1e-9 || abs(refs(4) - P.v_ref(k)) > 1e-9
        error('%s : consigne différente de Python (x_ref %g / %g, v_ref %g / %g).', ou, refs(3), P.x_ref(k), refs(4), P.v_ref(k));
    end
    [ex, ix] = max(abs(ox' - P.obs(k, :)) ./ max(1, abs(P.obs(k, :))));
    if ex > 1e-5, error('%s : composante %d de obs_x = %g (Python %g).', ou, ix, ox(ix), P.obs(k, ix)); end
    if abs(a_x - P.a(k)) > 1e-4, error('%s : action x %g (Python %g).', ou, a_x, P.a(k)); end
    u_th = -K * [P.theta(k) - 0.2 * a_x; P.wy(k); P.wdy(k); P.wdy_prev(k); P.U(k, :)'];
    u_th = min(max(u_th, -0.5), 0.5);
    if abs(u_th - P.u_theta(k)) > 1e-6, error('%s : u_theta du LQ %g (Python %g).', ou, u_th, P.u_theta(k)); end
end
clear controleur_x
fprintf('Vérification OK : controleur_x.m + LQ = Python sur %d pas (agent x et boucle interne).\n', numel(P.t));
end
