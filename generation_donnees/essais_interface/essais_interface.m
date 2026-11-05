% Étape A : impulsions différentielles pour trouver le mélangeur, les unités et les signes de UAV_model.
% Le modèle de test est construit par script et jamais sauvegardé. Résultats : resultats_essais_interface.mat
% (colonnes : t, 18 sorties nommées, 4 commandes moteurs, 18 valeurs de all_outputs).
S = fileparts(mfilename('fullpath')); cd(S); addpath(S); addpath(fullfile(S, '..'));
global MOTIF AMP T_ON T_OFF
mdl = 'test_melangeur';
if bdIsLoaded(mdl), close_system(mdl, 0); end
new_system(mdl); 
set_param(mdl, 'SolverType', 'Fixed-step', 'Solver', 'FixedStepAuto', 'FixedStep', 'Ts', 'StopTime', 'T_end');
add_block('simulink/Ports & Subsystems/Subsystem Reference', [mdl '/UAV'], 'ReferencedSubsystem', 'UAV_model');
add_block('simulink/Sources/Clock', [mdl '/Clock']);
add_block('simulink/Signal Routing/Mux', [mdl '/Mux'], 'Inputs', '2');
add_block('simulink/Discrete/Zero-Order Hold', [mdl '/ZOH'], 'SampleTime', 'Ts');
add_block('simulink/User-Defined Functions/Interpreted MATLAB Function', [mdl '/ctrl'], 'MATLABFcn', 'controleur_essais_interface(u)', 'OutputDimensions', '4', 'SampleTime', 'Ts');
add_block('simulink/Signal Routing/Demux', [mdl '/Demux'], 'Outputs', '4');
add_block('simulink/Sinks/To Workspace', [mdl '/log'], 'VariableName', 'logA', 'SaveFormat', 'Array', 'SampleTime', 'Ts');
add_block('simulink/Signal Routing/Mux', [mdl '/MuxLog'], 'Inputs', '3');
add_block('simulink/Signal Routing/Mux', [mdl '/MuxY'], 'Inputs', '18');
pU = get_param([mdl '/UAV'], 'PortHandles');
for k = 1:18, add_line(mdl, sprintf('UAV/%d', k), sprintf('MuxY/%d', k)); end
add_line(mdl, 'Clock/1', 'Mux/1'); add_line(mdl, 'MuxY/1', 'Mux/2');
add_block('simulink/Discrete/Zero-Order Hold', [mdl '/ZOH19'], 'SampleTime', 'Ts'); add_line(mdl, 'UAV/19', 'ZOH19/1');
add_line(mdl, 'Mux/1', 'ZOH/1'); add_line(mdl, 'ZOH/1', 'ctrl/1'); add_line(mdl, 'ctrl/1', 'Demux/1');
for k = 1:4, add_line(mdl, sprintf('Demux/%d', k), sprintf('UAV/%d', k)); end
add_line(mdl, 'ZOH/1', 'MuxLog/1'); add_line(mdl, 'ctrl/1', 'MuxLog/2'); add_line(mdl, 'ZOH19/1', 'MuxLog/3'); add_line(mdl, 'MuxLog/1', 'log/1');
random_init; Ts = 0.05; g_noise = 0;
% motifs candidats (moteurs 1..4) : celui du lacet vient de commande_LQ/test_LQI_psi.slx
essais_ = {'lacet', [1 -1 1 -1], 0.02, 3, 4, 12;
           'A',     [1 1 -1 -1], 0.005, 3, 3.3, 5;
           'B',     [1 -1 -1 1], 0.005, 3, 3.3, 5;
           'aucun', [0 0 0 0],   0,     3, 4, 6};
R = struct();
for i = 1:size(essais_, 1)
    [nom, MOTIF, AMP, T_ON, T_OFF, T_end] = essais_{i, :};
    clear controleur_essais_interface
    out = sim(mdl);
    R.(nom) = out.logA;              % colonnes : t, 18 sorties NOMMÉES, 4 commandes, 18 de all_outputs
    fprintf('%s fait (%d lignes)\n', nom, size(out.logA, 1));
end
save(fullfile(S, 'resultats_essais_interface.mat'), 'R');
close_system(mdl, 0);
