function verifier_xy(ici)
% VERIFIER_XY  Vérification pas à pas du contrôleur MATLAB de l'étape « xy » contre le vol Python
% exporté (traj_*.mat) : mêmes consignes, observations, actions et commandes. Appelé par EINT_RL_eval_xy.m
% et par ../verifier_tout.m. Argument : ici = dossier simulink_RL/xy/.
% Rejoue le vol Python (traj_xy.mat) dans controleur_xy.m : consignes, observations et actions des deux axes, puis
% les commandes u_theta et u_phi du LQ d'attitude, pas par pas.
P = load(fullfile(ici, 'traj_xy.mat')); S = load(fullfile(ici, 'lq_attitude.mat')); K = S.K_ATT(:)';
clear controleur_xy
for k = 1:numel(P.t)
    [~, ~, a_av, a_co, refs, o_av, o_co] = controleur_xy(P.t(k), 0.5, 0, 0, 0.39, P.psi(k), 0, 0, 0, P.x(k), P.y(k), ...
        P.vx(k), P.vy(k), P.theta(k), P.phi(k), P.wx(k), P.wy(k), 'rl');
    ou = sprintf('pas %d (t = %.2f s)', k - 1, P.t(k));
    if max(abs(refs(3:6)' - P.refs(k, :))) > 1e-9 || abs(refs(2) - P.psi_ref(k)) > 1e-9
        error('%s : consigne différente de Python.', ou);
    end
    for c = {{o_av, P.obs_av(k, :), 'avant'}, {o_co, P.obs_co(k, :), 'côté'}}
        [ex, ix] = max(abs(c{1}{1}' - c{1}{2}) ./ max(1, abs(c{1}{2})));
        if ex > 1e-5, error('%s : composante %d de l''observation %s = %g (Python %g).', ou, ix, c{1}{3}, c{1}{1}(ix), c{1}{2}(ix)); end
    end
    if abs(a_av - P.a_av(k)) > 1e-4 || abs(a_co - P.a_co(k)) > 1e-4
        error('%s : actions %g / %g (Python %g / %g).', ou, a_av, a_co, P.a_av(k), P.a_co(k));
    end
    u_th = min(max(-K * [P.theta(k) - 0.2 * a_av; P.wy(k); P.wdy(k); P.wdy_prev(k); P.Uth(k, :)'], -0.5), 0.5);
    u_ph = min(max(-K * [P.phi(k) + 0.2 * a_co; P.wx(k); P.wdx(k); P.wdx_prev(k); P.Uph(k, :)'], -0.5), 0.5);
    if abs(u_th - P.u_theta(k)) > 1e-6 || abs(u_ph - P.u_phi(k)) > 1e-6
        error('%s : u_theta / u_phi du LQ %g / %g (Python %g / %g).', ou, u_th, u_ph, P.u_theta(k), P.u_phi(k));
    end
end
clear controleur_xy
fprintf('Vérification OK : controleur_xy.m + LQ = Python sur %d pas (deux axes, rotation par psi).\n', numel(P.t));
end
