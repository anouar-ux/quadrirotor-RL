%% Validation du LQ d'attitude sur UAV_model (étape C1)
%  Le drone décolle à 0.5 m, tient psi = 0, et theta / phi suivent des échelons de consigne (controleur_LQ_ARX_attitude.m).
%  Deux vols : sans bruit et avec le bruit réaliste g_noise = 0.1. Comparaison à la réponse PRÉDITE par le modèle ARX.
%  Sorties : resultats_test_LQ_ARX_attitude.mat, vol_LQ_ARX_attitude.png
clear; clc;
here = fileparts(mfilename('fullpath')); cd(here); addpath(here); addpath(fullfile(here, '..'));
global K_ATT
L_ = load(fullfile(here, 'gains_LQ_ARX_attitude.mat')); K_ATT = L_.K;
Ts = 0.05; T_end = 10;
mdl = construire_modele();
V = struct();
for g = [0, 0.1]
    random_init; Ts = 0.05; g_noise = g;
    clear controleur_LQ_ARX_attitude
    out = sim(mdl);
    V.(sprintf('g%03d', round(1000 * g))) = out.logV;   % t, 18 sorties, u1..u4, theta_ref, phi_ref, theta_est
end
close_system(mdl, 0);

% Réponse prédite par le modèle ARX en boucle fermée, aux mêmes consignes
L = V.g000; t = L(:, 1); n = numel(t);
pred = zeros(n, 2);
for c = 1:2
    ref = L(:, 23 + c); x = zeros(9, 1);
    for k = 1:n
        pred(k, c) = x(1);
        xe = x; xe(1) = x(1) - ref(k);
        x = L_.A(1:9, 1:9) * x + L_.B(1:9) * (-K_ATT(1:9) * xe);
    end
end
save(fullfile(here, 'resultats_test_LQ_ARX_attitude.mat'), 'V', 'pred', 'K_ATT');

figure('Visible', 'off', 'Position', [100 100 1100 750]);
theme(gcf, 'light');              % figures en mode clair, même si Windows / MATLAB sont en mode sombre
noms = {'theta', 'phi'};
for c = 1:2
    subplot(3, 2, c); hold on;
    plot(t, L(:, 23 + c), 'k:', 'LineWidth', 1.2);
    plot(t, pred(:, c), 'Color', [0.6 0.6 0.6], 'LineWidth', 2.5);
    G = V.g100;
    if c == 1, a0 = L(:, 26); a1 = G(:, 26); else, a0 = L(:, 1 + 7); a1 = G(:, 1 + 7); end
    plot(t, a0, 'LineWidth', 1.2); plot(t, a1, 'LineWidth', 1);
    ylabel([noms{c} ' [rad]']); grid on; title(noms{c});
    if c == 1, legend('consigne', 'prédiction ARX', 'Simulink sans bruit', 'Simulink g\_noise = 0.1', 'Location', 'southwest'); end
    subplot(3, 2, 2 + c); hold on;
    du = [L(:, 22) - L(:, 20), L(:, 21) - L(:, 23)] / 2; dg = [G(:, 22) - G(:, 20), G(:, 21) - G(:, 23)] / 2;
    stairs(t, du(:, c)); stairs(t, dg(:, c)); ylabel(['u\_' noms{c}]); grid on;
end
subplot(3, 2, 5); plot(t, V.g000(:, 4), t, V.g100(:, 4)); ylabel('z [m]'); xlabel('t [s]'); grid on;
subplot(3, 2, 6); plot(t, V.g000(:, 2), t, V.g000(:, 3), t, V.g100(:, 2), '--', t, V.g100(:, 3), '--');
ylabel('position [m]'); xlabel('t [s]'); legend('x', 'y', 'x (bruit)', 'y (bruit)', 'Location', 'northwest'); grid on;
exportgraphics(gcf, fullfile(here, 'vol_LQ_ARX_attitude.png'), 'Resolution', 130);

for nom = fieldnames(V)'
    G = V.(nom{1}); i = t >= 3;
    eth = G(i, 26) - G(i, 24); eph = G(i, 1 + 7) - G(i, 25);
    fprintf('%s : écart à la prédiction ARX  theta %.4f rad, phi %.4f rad (RMS) ; z dans [%.3f, %.3f] ; |psi| max %.3f rad\n', ...
        nom{1}, rms(G(i, 26) - pred(i, 1)), rms(G(i, 1 + 7) - pred(i, 2)), min(G(i, 4)), max(G(i, 4)), max(abs(G(:, 10))));
end

function mdl = construire_modele()
% UAV_model -> 18 sorties nommées -> ZOH Ts -> controleur_LQ_ARX_attitude (7 sorties) -> 4 moteurs ; journal : t, 18, 7
mdl = 'boucle_LQ_ARX_attitude';
if bdIsLoaded(mdl), close_system(mdl, 0); end
new_system(mdl);
set_param(mdl, 'SolverType', 'Fixed-step', 'Solver', 'FixedStepAuto', 'FixedStep', 'Ts', 'StopTime', 'T_end');
add_block('simulink/Ports & Subsystems/Subsystem Reference', [mdl '/UAV'], 'ReferencedSubsystem', 'UAV_model');
add_block('simulink/Sources/Clock', [mdl '/Clock']);
add_block('simulink/Signal Routing/Mux', [mdl '/MuxY'], 'Inputs', '18');
add_block('simulink/Signal Routing/Mux', [mdl '/Mux'], 'Inputs', '2');
add_block('simulink/Discrete/Zero-Order Hold', [mdl '/ZOH'], 'SampleTime', 'Ts');
add_block('simulink/User-Defined Functions/Interpreted MATLAB Function', [mdl '/ctrl'], ...
    'MATLABFcn', 'controleur_LQ_ARX_attitude(u)', 'OutputDimensions', '7', 'SampleTime', 'Ts');
add_block('simulink/Signal Routing/Demux', [mdl '/Demux'], 'Outputs', '[4 3]');
add_block('simulink/Signal Routing/Demux', [mdl '/DemuxU'], 'Outputs', '4');
add_block('simulink/Signal Routing/Mux', [mdl '/MuxLog'], 'Inputs', '2');
add_block('simulink/Sinks/To Workspace', [mdl '/log'], 'VariableName', 'logV', 'SaveFormat', 'Array', 'SampleTime', 'Ts');
add_block('simulink/Sinks/Terminator', [mdl '/T19']); add_line(mdl, 'UAV/19', 'T19/1');
add_block('simulink/Sinks/Terminator', [mdl '/Tref']);
for k = 1:18, add_line(mdl, sprintf('UAV/%d', k), sprintf('MuxY/%d', k)); end
add_line(mdl, 'Clock/1', 'Mux/1'); add_line(mdl, 'MuxY/1', 'Mux/2'); add_line(mdl, 'Mux/1', 'ZOH/1');
add_line(mdl, 'ZOH/1', 'ctrl/1'); add_line(mdl, 'ctrl/1', 'Demux/1');
add_line(mdl, 'Demux/1', 'DemuxU/1'); add_line(mdl, 'Demux/2', 'Tref/1');
for k = 1:4, add_line(mdl, sprintf('DemuxU/%d', k), sprintf('UAV/%d', k)); end
add_line(mdl, 'ZOH/1', 'MuxLog/1'); add_line(mdl, 'ctrl/1', 'MuxLog/2'); add_line(mdl, 'MuxLog/1', 'log/1');
end
