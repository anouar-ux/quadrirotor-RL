%% Génération des données d'identification du TANGAGE (θ) et du ROULIS (φ) sur UAV_model (étape C1)
%  Même démarche que generation_lacet/generation_donnees_psi.m : décollage à 0.5 m (LQI z), lacet tenu à 0, puis
%  excitation SIMULTANÉE et INDÉPENDANTE de u_theta et u_phi (deux signaux multi-niveaux tirés séparément)
%  pendant DUREE secondes, avec un PD faible sur chaque angle pour que le drone reste presque à plat.
%  Sortie : EI_RL/donnees/attitude/donnees_attitude.mat, un essai par ligne, depuis le début de l'excitation :
%    commandes appliquées (u_theta, u_phi, u_psi, u_z) et mesures (angles, vitesses et accélérations angulaires,
%    positions, vitesses et accélérations x, y, z) : de quoi identifier aussi x et y pour l'étape C2.
%  theta est l'intégrale de wy (la sortie theta de UAV_model est noyée dans le bruit, voir controleur_generation_attitude.m) ;
%  la sortie brute est gardée dans theta_mesure.
clear; clc;
here = fileparts(mfilename('fullpath')); cd(here); addpath(here); addpath(fullfile(here, '..'));   % UAV_model.slx, random_init.m
global EXC_THETA EXC_PHI K_ANG K_W T_DEBUT
N_ESSAIS = str2double(getenv_def('N_ESSAIS', '150'));
DUREE = str2double(getenv_def('DUREE', '60'));   % s d'excitation par essai
SORTIE = getenv_def('SORTIE', fullfile(here, '..', '..', 'donnees', 'attitude', 'donnees_attitude.mat'));
T_DEBUT = 3;                  % s : décollage et stabilisation de z avant l'excitation
K_ANG = 0.0235;               % PD d'attitude faible : theta'' = 170 u -> pulsation ~2 rad/s,
K_W = 0.0165;                 %   amortissement ~0.7 (bande passante bien en dessous du retard moteur ~0.2 s)
AMP_MAX = 0.006;              % excitation max : angle établi ~ AMP_MAX / K_ANG ~ 0.25 rad
G_NOISE = 0.1;                % bruit réaliste
Ts = 0.05; T_end = T_DEBUT + DUREE;

mdl = construire_modele();
rng(2027);
n = round(DUREE / Ts);
noms = {'u_theta', 'u_phi', 'u_psi', 'u_z', 'theta', 'phi', 'psi', 'wy', 'wx', 'wz', 'wdy', 'wdx', ...
        'x', 'y', 'z', 'vx', 'vy', 'vz', 'ax', 'ay', 'az'};
col = struct('x', 1, 'y', 2, 'z', 3, 'vx', 4, 'vy', 5, 'vz', 6, 'phi', 7, 'theta', 8, 'psi', 9, ...
             'wx', 10, 'wy', 11, 'wz', 12, 'ax', 13, 'ay', 14, 'az', 15, 'wdx', 16, 'wdy', 17);   % sorties nommées
D = struct();
for j = 1:numel(noms), D.(noms{j}) = zeros(N_ESSAIS, n); end
D.theta_mesure = zeros(N_ESSAIS, n);
D.amp = zeros(N_ESSAIS, 2);
t0 = tic;
for i = 1:N_ESSAIS
    amp = 0.0005 + (AMP_MAX - 0.0005) * rand(1, 2);      % amplitudes max de l'essai (theta, phi)
    EXC_THETA = signal_multiniveaux(n, amp(1));
    EXC_PHI = signal_multiniveaux(n, amp(2));
    random_init; Ts = 0.05; g_noise = G_NOISE;
    clear controleur_generation_attitude
    out = sim(mdl);
    L = out.logC;                                         % t, 18 sorties, 4 commandes
    k = find(abs(L(:, 1) - T_DEBUT) < 1e-6) + (0:n - 1);  % pas d'excitation
    U = L(k, 20:23);
    D.u_theta(i, :) = (U(:, 3) - U(:, 1)) / 2;            % commandes RÉELLEMENT appliquées (après saturation)
    D.u_phi(i, :) = (U(:, 2) - U(:, 4)) / 2;
    D.u_psi(i, :) = (U(:, 1) - U(:, 2) + U(:, 3) - U(:, 4)) / 4;
    D.u_z(i, :) = mean(U, 2);
    for j = 5:numel(noms), D.(noms{j})(i, :) = L(k, 1 + col.(noms{j})); end
    D.theta_mesure(i, :) = D.theta(i, :);                 % sortie theta de UAV_model : bruit ~30 rad (Cov = seed_theta)
    th = cumtrapz(L(:, 1), L(:, 1 + col.wy));             % theta ESTIMÉ comme dans controleur_generation_attitude.m (intégrale de wy)
    D.theta(i, :) = th(k);
    D.amp(i, :) = amp;
    if mod(i, 10) == 0 || i == 1
        fprintf('essai %3d / %d  (amp %.4f %.4f, |theta| max %.2f, |phi| max %.2f rad, z dans [%.3f, %.3f], |x| max %.1f m)  %.0f s\n', ...
            i, N_ESSAIS, amp, max(abs(D.theta(i, :))), max(abs(D.phi(i, :))), min(D.z(i, :)), max(D.z(i, :)), ...
            max(abs(D.x(i, :))), toc(t0));
    end
end
D.Ts = Ts; D.K_ANG = K_ANG; D.K_W = K_W; D.g_noise = G_NOISE; D.description = ...
    'attitude : u(k) appliqué au pas k, mesures au pas k (ZOH), depuis le début de l''excitation';
if ~exist(fileparts(SORTIE), 'dir'), mkdir(fileparts(SORTIE)); end
save(SORTIE, '-struct', 'D');
close_system(mdl, 0);
fprintf('%s écrit : %d essais x %d pas, %.0f s\n', SORTIE, N_ESSAIS, n, toc(t0));

function e = signal_multiniveaux(n, amp)
% Paliers de niveau uniforme dans [-amp, amp], maintenus 1 à 40 pas (0.05 à 2 s) : basses et hautes fréquences
e = zeros(n, 1); k = 1;
while k <= n
    d = randi([1, 40]); e(k:min(n, k + d - 1)) = amp * (2 * rand() - 1); k = k + d;
end
end

function v = getenv_def(nom, defaut)
v = getenv(nom); if isempty(v), v = defaut; end
end

function mdl = construire_modele()
% Boucle d'essai (non sauvegardée) : UAV_model -> 18 sorties nommées -> ZOH Ts -> controleur_generation_attitude -> 4 moteurs
mdl = 'boucle_attitude';
if bdIsLoaded(mdl), close_system(mdl, 0); end
new_system(mdl);
set_param(mdl, 'SolverType', 'Fixed-step', 'Solver', 'FixedStepAuto', 'FixedStep', 'Ts', 'StopTime', 'T_end');
add_block('simulink/Ports & Subsystems/Subsystem Reference', [mdl '/UAV'], 'ReferencedSubsystem', 'UAV_model');
add_block('simulink/Sources/Clock', [mdl '/Clock']);
add_block('simulink/Signal Routing/Mux', [mdl '/MuxY'], 'Inputs', '18');
add_block('simulink/Signal Routing/Mux', [mdl '/Mux'], 'Inputs', '2');
add_block('simulink/Discrete/Zero-Order Hold', [mdl '/ZOH'], 'SampleTime', 'Ts');
add_block('simulink/User-Defined Functions/Interpreted MATLAB Function', [mdl '/ctrl'], ...
    'MATLABFcn', 'controleur_generation_attitude(u)', 'OutputDimensions', '4', 'SampleTime', 'Ts');
add_block('simulink/Signal Routing/Demux', [mdl '/Demux'], 'Outputs', '4');
add_block('simulink/Signal Routing/Mux', [mdl '/MuxLog'], 'Inputs', '2');
add_block('simulink/Sinks/To Workspace', [mdl '/log'], 'VariableName', 'logC', 'SaveFormat', 'Array', 'SampleTime', 'Ts');
add_block('simulink/Sinks/Terminator', [mdl '/T19']); add_line(mdl, 'UAV/19', 'T19/1');
for k = 1:18, add_line(mdl, sprintf('UAV/%d', k), sprintf('MuxY/%d', k)); end
add_line(mdl, 'Clock/1', 'Mux/1'); add_line(mdl, 'MuxY/1', 'Mux/2'); add_line(mdl, 'Mux/1', 'ZOH/1');
add_line(mdl, 'ZOH/1', 'ctrl/1'); add_line(mdl, 'ctrl/1', 'Demux/1');
for k = 1:4, add_line(mdl, sprintf('Demux/%d', k), sprintf('UAV/%d', k)); end
add_line(mdl, 'ZOH/1', 'MuxLog/1'); add_line(mdl, 'ctrl/1', 'MuxLog/2'); add_line(mdl, 'MuxLog/1', 'log/1');
end
