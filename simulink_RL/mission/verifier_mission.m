function verifier_mission(ici)
% VERIFIER_MISSION  Vérification pas à pas du contrôleur MATLAB de l'étape « mission » contre le vol Python
% exporté (traj_*.mat) : mêmes consignes, observations, actions et commandes. Appelé par EINT_RL_eval_mission.m
% et par ../verifier_tout.m. Argument : ici = dossier simulink_RL/mission/.
% Rejoue les entrées v du vol Python dans rl_controller_mission.m et compare les sorties y, pas par pas.
global MODE_MISSION
MODE_MISSION = 'rl';
P = load(fullfile(ici, 'traj_mission_cap.mat'));
clear rl_controller_mission
noms = {'u1', 'u2', 'u3', 'u4', 'phase', 'z_ref', 'psi_ref', 'x_ref', 'y_ref', 'x_v', 'y_v', 'theta_ref', 'phi_ref', ...
        'u_z', 'u_theta', 'u_phi'};
for k = 1:size(P.v, 1)
    y = rl_controller_mission(P.v(k, :)');
    [ex, ix] = max(abs(y' - P.y(k, :)));
    if ex > 1e-6
        error('pas %d (t = %.2f s) : %s = %g (Python %g).', k - 1, P.v(k, 1), noms{ix}, y(ix), P.y(k, ix));
    end
end
clear rl_controller_mission
fprintf('Vérification OK : rl_controller_mission.m = Python sur %d pas (superviseur, 3 agents, LQ, mélangeur).\n', ...
    size(P.v, 1));
end
