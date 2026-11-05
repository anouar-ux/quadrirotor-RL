function mdl = construire_boucle_xy()
%CONSTRUIRE_BOUCLE_XY  Crée et enregistre EINT_RL_closed_loop_xy.slx (à côté de ce fichier), comme construire_boucle_x.m (étape C3) :
%   UAV_model -> 18 sorties nommées -> ZOH Ts (avec l'horloge) -> rl_controller_xy -> 4 moteurs.
%   Journal rl_log_xy : [t, 18 sorties, u1..u4, z_ref, psi_ref, x_ref, y_ref, theta_ref, phi_ref, u_z_agent, u_theta, u_phi] à chaque pas.
ici = fileparts(mfilename('fullpath'));
addpath(ici, fullfile(ici, '..', '..', 'Matlab files'));
mdl = 'EINT_RL_closed_loop_xy';
NY = 13;                                   % nombre de sorties de rl_controller_xy
if bdIsLoaded(mdl), close_system(mdl, 0); end
new_system(mdl);
set_param(mdl, 'SolverType', 'Fixed-step', 'Solver', 'FixedStepAuto', 'FixedStep', 'Ts', 'StopTime', 'T_end');
add_block('simulink/Ports & Subsystems/Subsystem Reference', [mdl '/UAV'], 'ReferencedSubsystem', 'UAV_model', ...
    'Position', [300 80 420 520]);
add_block('simulink/Sources/Clock', [mdl '/Clock'], 'Position', [480 30 500 50]);
add_block('simulink/Signal Routing/Mux', [mdl '/MuxY'], 'Inputs', '18', 'Position', [480 80 485 520]);
add_block('simulink/Signal Routing/Mux', [mdl '/Mux'], 'Inputs', '2', 'Position', [540 40 545 300]);
add_block('simulink/Discrete/Zero-Order Hold', [mdl '/ZOH'], 'SampleTime', 'Ts', 'Position', [580 150 610 180]);
add_block('simulink/User-Defined Functions/Interpreted MATLAB Function', [mdl '/rl_controller_xy'], ...
    'MATLABFcn', 'rl_controller_xy(u)', 'OutputDimensions', num2str(NY), 'SampleTime', 'Ts', 'Position', [650 145 800 185]);
add_block('simulink/Signal Routing/Demux', [mdl '/Demux'], 'Outputs', num2str(NY), 'Position', [840 60 845 300]);
add_block('simulink/Signal Routing/Mux', [mdl '/MuxLog'], 'Inputs', '2', 'Position', [880 330 885 390]);
add_block('simulink/Sinks/To Workspace', [mdl '/rl_log_xy'], 'VariableName', 'rl_log_xy', ...
    'SaveFormat', 'Array', 'SampleTime', 'Ts', 'Position', [920 345 1000 375]);
add_block('simulink/Sinks/Terminator', [mdl '/T19'], 'Position', [480 560 500 580]);
add_line(mdl, 'UAV/19', 'T19/1');
for k = 1:18, add_line(mdl, sprintf('UAV/%d', k), sprintf('MuxY/%d', k)); end
add_line(mdl, 'Clock/1', 'Mux/1'); add_line(mdl, 'MuxY/1', 'Mux/2'); add_line(mdl, 'Mux/1', 'ZOH/1');
add_line(mdl, 'ZOH/1', 'rl_controller_xy/1'); add_line(mdl, 'rl_controller_xy/1', 'Demux/1');
for k = 1:4, add_line(mdl, sprintf('Demux/%d', k), sprintf('UAV/%d', k), 'autorouting', 'on'); end
for k = 5:NY
    add_block('simulink/Sinks/Terminator', sprintf('%s/T%d', mdl, k), 'Position', [880 40 + 25 * k 900 55 + 25 * k]);
    add_line(mdl, sprintf('Demux/%d', k), sprintf('T%d/1', k));
end
add_line(mdl, 'ZOH/1', 'MuxLog/1'); add_line(mdl, 'rl_controller_xy/1', 'MuxLog/2');
add_line(mdl, 'MuxLog/1', 'rl_log_xy/1');
save_system(mdl, fullfile(ici, [mdl '.slx']));
close_system(mdl, 0);
end
