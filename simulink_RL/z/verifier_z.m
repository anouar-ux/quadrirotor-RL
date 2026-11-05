function verifier_z(here)
% VERIFIER_Z  Copie conforme de check_controller (rl_interface/EINT_RL_eval.m, fourni), appelable sur n'importe quel
% dossier : rejoue traj_model.mat (vol Python de l'agent z) dans rl_controller.m et vérifie consignes, observations et
% commandes. Appelé par vols_z.m.
cd(here);
T = load(fullfile(here, 'traj_model.mat'));
clear rl_controller
for k = 1:numel(T.t)
    [u, z_ref, obs] = rl_controller(T.t(k), T.z(k), T.vz(k), T.az_prev(k), T.u_prev(k));
    where = sprintf('step %d (t = %.2f s)', k - 1, T.t(k));
    if abs(z_ref - T.z_ref(k)) > 1e-9
        error('%s: z_ref = %g in rl_controller.m, %g in Python: mission() differs.', where, z_ref, T.z_ref(k));
    end
    if numel(obs) ~= size(T.obs, 2)
        error('observation() returns %d values in rl_controller.m, %d in Python.', numel(obs), size(T.obs, 2));
    end
    [e, i] = max(abs(obs(:)' - T.obs(k, :)) ./ max(1, abs(T.obs(k, :))));
    if e > 1e-5
        error('%s: component %d of the observation is %g in rl_controller.m, %g in Python.', where, i, obs(i), T.obs(k, i));
    end
    if abs(u - T.u(k)) > 1e-4
        error('%s: u = %g in MATLAB, %g in Python, with the same observation: export again.', where, u, T.u(k));
    end
end
clear rl_controller
fprintf('Check OK: rl_controller.m flies the policy as in Python (%d steps of traj_model.mat).\n', numel(T.t));
end
