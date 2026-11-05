%% VOLS SIMULINK DE L'AGENT z (étape de base de la piste RL) : mission de la Figure 1 du sujet (0 -> 0,5 -> -0,25 m)
%  À quoi sert ce script : vérifier que rl_controller.m (rl_interface/, fichier fourni complété) pilote l'agent PPO z
%  exactement comme en Python (verifier_z.m), puis faire voler la mission sur EINT_RL_closed_loop.slx (fourni) sans bruit
%  (1 vol), avec g_noise = 0,1 (10 vols) et g_noise = 0,2 (5 vols). Les fichiers de rl_interface/ sont lus, jamais modifiés.
%  Avant : python python/export/export_z.py (écrit policy.mat et traj_model.mat dans rl_interface/).
%  Sortie : resultats_z.mat (dans ce dossier) ; figures et bilan : python simulink_RL/z/graphes_z.py
S = fileparts(mfilename('fullpath'));
here = fullfile(S, '..', '..', 'rl_interface');
addpath(S); addpath(fullfile(S, '..', '..', 'Matlab files'));
verifier_z(here);
cd(here); addpath(here);
niveaux = [0 0.1 0.2]; nrep = [1 10 5];
R = struct('g', {}, 'L', {});
for i = 1:numel(niveaux)
    for r = 1:nrep(i)
        random_init;
        Ts = 0.05; T_end = 10; g_noise = niveaux(i);
        clear rl_controller
        out = sim('EINT_RL_closed_loop');
        L = out.rl_log;
        R(end + 1) = struct('g', g_noise, 'L', L);
        fprintf('g_noise = %4.2f vol %2d : RMSE %.4f m, z min %.3f, z final %.4f\n', g_noise, r, ...
            sqrt(mean((L(:, 6) - L(:, 2)).^2)), min(L(:, 2)), L(end, 2));
    end
end
clear rl_controller
save(fullfile(S, 'resultats_z.mat'), 'R');
