%% RÉPONSES DES LQI « DOUBLE INTÉGRATEUR » SUR SIMULINK (référence pour le réajustement des gains à 20 Hz)
%  À quoi sert ce script : simuler sans bruit les trois modèles de test, où le LQI est CONTINU
%  (solveur à pas variable, pas de bloqueur d'ordre zéro), et enregistrer, sur une grille de 0,01 s :
%    test_LQI_z.slx   : z et sa consigne (échelon de 0,5 m à t = 5 s) ;
%    test_LQI_psi.slx : psi et sa consigne (rampes filtrées vues par PID_psi), z. Les rampes d'origine (18 rad/s
%                       pendant 10 s, soit 180 rad) sortent de la plage du modèle de lacet identifié : elles sont
%                       remplacées EN MÉMOIRE (le .slx n'est pas modifié) par le scan des missions, un tour
%                       en 8 s entre 5 et 13 s (pente 2 pi / 8), toujours filtré par 1 / (0,4 s + 1) ;
%    test_LQI_x.slx   : theta et theta_ref (sortie de la boucle de position), x et x_ref.
%  Ces réponses servent de CIBLE à python/identification/reajuster_gains_LQI.py, qui recale les gains (même structure
%  de LQI) pour un contrôleur échantillonné à 20 Hz, comme celui de l'interface RL et des comparaisons Python.
%  Sortie : reponses_LQI_simulink.mat (dans ce dossier).
run(fullfile(fileparts(mfilename('fullpath')), 'gains_LQI.m'));          % gains d'origine (contient clear)
ici = fileparts(which('gains_LQI'));
g_noise = 0;                                                              % sans bruit : réponse « nominale »
cas = {'test_LQI_z',   {'Subsystem Reference', 3, 'z'; 'Step3', 1, 'z_ref'};
       'test_LQI_psi', {'Subsystem Reference', 9, 'psi'; 'Transfer Fcn', 1, 'psi_ref'; 'Subsystem Reference', 3, 'z'};
       'test_LQI_x',   {'Subsystem Reference', 8, 'theta'; 'Saturation5', 1, 'theta_ref'; 'Subsystem Reference', 1, 'x'; ...
                        'Transfer Fcn', 1, 'x_ref'}};
REP = struct();
for c = 1:size(cas, 1)
    mdl = cas{c, 1}; sig = cas{c, 2};
    load_system(fullfile(ici, [mdl '.slx']));
    if strcmp(mdl, 'test_LQI_psi')                     % consigne du scan, en mémoire seulement
        set_param([mdl '/Ramp'], 'slope', num2str(2 * pi / 8), 'start', '5');
        set_param([mdl '/Ramp1'], 'slope', num2str(-2 * pi / 8), 'start', '13');
        set_param(mdl, 'StopTime', '20');
    end
    for i = 1:size(sig, 1)
        ph = get_param([mdl '/' sig{i, 1}], 'PortHandles');
        set_param(ph.Outport(sig{i, 2}), 'DataLogging', 'on', 'DataLoggingNameMode', 'Custom', 'DataLoggingName', sig{i, 3});
    end
    o = sim(mdl, 'SignalLogging', 'on', 'SignalLoggingName', 'lg');
    T = str2double(get_param(mdl, 'StopTime')); t = (0:0.01:T)';
    S = struct('t', t);
    for i = 1:size(sig, 1)
        e = o.lg.getElement(sig{i, 3});
        S.(sig{i, 3}) = interp1(e.Values.Time, double(squeeze(e.Values.Data)), t, 'linear', 'extrap');
    end
    REP.(mdl) = S;
    close_system(mdl, 0);                              % sans sauvegarde
    fprintf('%s : %d échantillons enregistrés\n', mdl, numel(t));
end
gains = struct('z', [Kp Kd Ki], 'psi', [Kp_psi Kd_psi Ki_psi], 'theta', [Kp_theta Kd_theta Ki_theta], 'x', [Kp_x Kd_x], 'tau', tau);
save(fullfile(ici, 'reponses_LQI_simulink.mat'), 'REP', 'gains');
fprintf('-> %s\n', fullfile(ici, 'reponses_LQI_simulink.mat'));
