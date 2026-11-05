%% Génération des données d'identification du LACET sur UAV_model (étape B1)
%  Le drone décolle et tient 0.5 m (LQI z), puis on excite u_psi pendant DUREE secondes avec un signal
%  multi-niveaux aléatoire (amplitudes et durées de maintien tirées au hasard), avec bruit de mesure réaliste.
%  Sortie : EI_RL/donnees/lacet/donnees_lacet.mat, un essai par ligne, à partir du début de l'excitation (drone au repos en lacet) :
%    u_psi (commande différentielle appliquée), wdz (accélération de lacet mesurée), wz, psi, z, u_z
clear; clc;
here = fileparts(mfilename('fullpath')); cd(here); addpath(here); addpath(fullfile(here, '..'));   % UAV_model.slx, random_init.m
global EXCITATION K_W T_DEBUT
N_ESSAIS = str2double(getenv_def('N_ESSAIS', '150'));
DUREE = 60;                   % s d'excitation par essai
T_DEBUT = 3;                  % s : décollage et stabilisation de z avant l'excitation
K_W = 0.006;                  % amortissement de wz : constante de temps ~ 1 / (170 * K_W) ~ 1 s
G_NOISE = 0.1;                % bruit réaliste (étape A : proche des données réelles)
Ts = 0.05; T_end = T_DEBUT + DUREE;

mdl = construire_modele();
rng(2026);                    % excitations reproductibles (le bruit du simulateur, lui, vient de random_init)
n = round(DUREE / Ts);
D = struct('u_psi', zeros(N_ESSAIS, n), 'wdz', zeros(N_ESSAIS, n), 'wz', zeros(N_ESSAIS, n), ...
           'psi', zeros(N_ESSAIS, n), 'z', zeros(N_ESSAIS, n), 'u_z', zeros(N_ESSAIS, n), 'amp', zeros(N_ESSAIS, 1));
t0 = tic;
for i = 1:N_ESSAIS
    amp = 0.002 + 0.028 * rand();                         % amplitude max de l'essai : 0.002 à 0.03
    EXCITATION = signal_multiniveaux(n, amp);
    random_init; Ts = 0.05; g_noise = G_NOISE;
    clear controleur_generation_psi
    out = sim(mdl);
    L = out.logL;                                         % t, 18 sorties, 4 commandes
    k = find(abs(L(:, 1) - T_DEBUT) < 1e-6) + (0:n - 1);  % pas d'excitation
    U = L(k, 20:23);
    D.u_psi(i, :) = (U(:, 1) - U(:, 2) + U(:, 3) - U(:, 4)) / 4;   % commande de lacet RÉELLEMENT appliquée
    D.u_z(i, :) = mean(U, 2);
    D.wdz(i, :) = L(k, 1 + 18); D.wz(i, :) = L(k, 1 + 12); D.psi(i, :) = L(k, 1 + 9); D.z(i, :) = L(k, 1 + 3);
    D.amp(i) = amp;
    if mod(i, 10) == 0 || i == 1
        fprintf('essai %3d / %d  (amp %.3f, |wz| max %.2f rad/s, z dans [%.3f, %.3f])  %.0f s\n', i, N_ESSAIS, amp, ...
            max(abs(D.wz(i, :))), min(D.z(i, :)), max(D.z(i, :)), toc(t0));
    end
end
D.Ts = Ts; D.K_W = K_W; D.g_noise = G_NOISE; D.description = ...
    'lacet : u_psi(k) appliqué au pas k, wdz/wz/psi/z mesurés au pas k (ZOH), depuis le début de l''excitation';
save(fullfile(here, '..', '..', 'donnees', 'lacet', 'donnees_lacet.mat'), '-struct', 'D');
close_system(mdl, 0);
fprintf('donnees_lacet.mat écrit : %d essais x %d pas, %.0f s\n', N_ESSAIS, n, toc(t0));

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
% Boucle d'essai (non sauvegardée) : UAV_model -> 18 sorties nommées -> ZOH Ts -> controleur_generation_psi -> 4 moteurs
mdl = 'boucle_lacet';
if bdIsLoaded(mdl), close_system(mdl, 0); end
new_system(mdl);
set_param(mdl, 'SolverType', 'Fixed-step', 'Solver', 'FixedStepAuto', 'FixedStep', 'Ts', 'StopTime', 'T_end');
add_block('simulink/Ports & Subsystems/Subsystem Reference', [mdl '/UAV'], 'ReferencedSubsystem', 'UAV_model');
add_block('simulink/Sources/Clock', [mdl '/Clock']);
add_block('simulink/Signal Routing/Mux', [mdl '/MuxY'], 'Inputs', '18');
add_block('simulink/Signal Routing/Mux', [mdl '/Mux'], 'Inputs', '2');
add_block('simulink/Discrete/Zero-Order Hold', [mdl '/ZOH'], 'SampleTime', 'Ts');
add_block('simulink/User-Defined Functions/Interpreted MATLAB Function', [mdl '/ctrl'], ...
    'MATLABFcn', 'controleur_generation_psi(u)', 'OutputDimensions', '4', 'SampleTime', 'Ts');
add_block('simulink/Signal Routing/Demux', [mdl '/Demux'], 'Outputs', '4');
add_block('simulink/Signal Routing/Mux', [mdl '/MuxLog'], 'Inputs', '2');
add_block('simulink/Sinks/To Workspace', [mdl '/log'], 'VariableName', 'logL', 'SaveFormat', 'Array', 'SampleTime', 'Ts');
add_block('simulink/Sinks/Terminator', [mdl '/T19']); add_line(mdl, 'UAV/19', 'T19/1');
for k = 1:18, add_line(mdl, sprintf('UAV/%d', k), sprintf('MuxY/%d', k)); end
add_line(mdl, 'Clock/1', 'Mux/1'); add_line(mdl, 'MuxY/1', 'Mux/2'); add_line(mdl, 'Mux/1', 'ZOH/1');
add_line(mdl, 'ZOH/1', 'ctrl/1'); add_line(mdl, 'ctrl/1', 'Demux/1');
for k = 1:4, add_line(mdl, sprintf('Demux/%d', k), sprintf('UAV/%d', k)); end
add_line(mdl, 'ZOH/1', 'MuxLog/1'); add_line(mdl, 'ctrl/1', 'MuxLog/2'); add_line(mdl, 'MuxLog/1', 'log/1');
end
