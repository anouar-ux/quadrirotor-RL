% Vols Simulink de la tâche avancée (z + lacet) : 1 vol sans bruit, 5 vols à g_noise = 0,1 ; rejeu hors ligne des mesures
% pour vérifier que le contrôleur a été appelé une fois par pas. Résultats : resultats_zpsi.mat
ici = fileparts(mfilename('fullpath')); cd(ici); addpath(ici, fullfile(ici, '..', '..', 'Matlab files'));
niveaux = [0 0.1]; nrep = [1 5];
R = struct('g', {}, 'L', {}, 'ecart_rejeu', {});
for i = 1:numel(niveaux)
    for r = 1:nrep(i)
        random_init; Ts = 0.05; T_end = 20; g_noise = niveaux(i);
        clear rl_controller_zpsi controleur_zpsi
        out = sim('EINT_RL_closed_loop_zpsi'); L = out.rl_log_zpsi;
        clear rl_controller_zpsi controleur_zpsi
        yr = zeros(size(L, 1), 6);
        for k = 1:size(L, 1), yr(k, :) = rl_controller_zpsi(L(k, 1:19)')'; end
        clear rl_controller_zpsi controleur_zpsi
        R(end + 1) = struct('g', g_noise, 'L', L, 'ecart_rejeu', max(abs(yr(:, 1:4) - L(:, 20:23)), [], 'all'));
        fprintf('g_noise %.2f vol %d : RMSE z %.4f m, RMSE psi %.4f rad, psi final %.4f (2 pi = %.4f), z fin %.3f, rejeu %.1e\n', ...
            g_noise, r, sqrt(mean((L(:, 24) - L(:, 4)).^2)), sqrt(mean((L(:, 25) - L(:, 10)).^2)), L(end, 10), 2 * pi, L(end, 4), R(end).ecart_rejeu);
    end
end
save(fullfile(ici, 'resultats_zpsi.mat'), 'R');
