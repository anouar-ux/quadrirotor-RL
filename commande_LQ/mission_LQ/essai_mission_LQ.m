%% ESSAI DE LA MISSION LQ : décollage à 0,5 m -> scan 360° -> poursuite -> posé sur la voiture (rectiligne uniforme)
%  À quoi sert ce script : lancer parametres_mission_LQ.m (gains du LQI + voiture), simuler mission_LQ.slx
%  pour 3 niveaux de bruit (g_noise = 0, 0,01, 0,1), afficher le bilan (instant du posé, écarts, vitesse relative) et
%  tracer la figure. Sorties : resultats_mission_LQ.mat, mission_LQ.png (dans ce dossier).
run(fullfile(fileparts(mfilename('fullpath')), 'parametres_mission_LQ.m'));   % contient clear
dossier = fileparts(mfilename('fullpath'));
g_defaut = g_noise;
mdl = 'mission_LQ'; load_system(fullfile(dossier, [mdl '.slx']));

% --- signaux enregistrés (en mémoire, le modèle n'est pas sauvegardé)
ph = get_param([mdl '/Subsystem Reference'], 'PortHandles');
pm = get_param([mdl '/mission'], 'PortHandles');
ps = get_param([mdl '/Saturation6'], 'PortHandles');
lg = {ph.Outport(1), 'x'; ph.Outport(2), 'y'; ph.Outport(3), 'z'; ph.Outport(4), 'vx'; ph.Outport(9), 'psi'; ...
      pm.Outport(1), 'x_ref'; pm.Outport(6), 'phase'; pm.Outport(7), 'x_v'; ps.Outport(1), 'theta_ref'};
for i = 1:size(lg, 1), set_param(lg{i, 1}, 'DataLogging', 'on', 'DataLoggingNameMode', 'Custom', 'DataLoggingName', lg{i, 2}); end

bruits = unique([0 g_defaut 0.1]);
R = struct();
for b = 1:numel(bruits)
    g_noise = bruits(b);
    o = sim(mdl, 'SignalLogging', 'on', 'SignalLoggingName', 'lg');
    t = (0:0.05:50)';
    v = @(n) interp1(o.lg.getElement(n).Values.Time, double(squeeze(o.lg.getElement(n).Values.Data)), t, 'previous', 'extrap');
    S = struct('t', t); for i = 1:size(lg, 1), S.(lg{i, 2}) = v(lg{i, 2}); end
    R(b).g = g_noise; R(b).S = S;
    k4 = find(S.phase >= 4, 1); k5 = find(S.phase >= 5, 1);
    fprintf('\n=== g_noise = %g ===\n', g_noise);
    fprintf('psi max %.3f rad (2pi = %.3f) | derive max pendant le scan : x %.3f m, y %.3f m\n', max(S.psi), 2*pi, max(abs(S.x(t < 15))), max(abs(S.y(t < 15))));
    if ~isempty(k4), fprintf('alignement au-dessus de la voiture a t = %.2f s\n', t(k4)); end
    if ~isempty(k5)
        fprintf('POSE a t = %.2f s : ecart x = %.3f m, ecart y = %.3f m, vitesse relative = %.3f m/s\n', ...
            t(k5), S.x(k5) - S.x_v(k5), S.y(k5), S.vx(k5) - v_v);
    else
        fprintf('pas de pose avant 50 s (ecart final %.3f m)\n', S.x(end) - S.x_v(end));
    end
end
save(fullfile(dossier, 'resultats_mission_LQ.mat'), 'R');

% --- figure
f = figure('Position', [0 0 1100 800]);
theme(f, 'light');                % figures en mode clair, même si Windows / MATLAB sont en mode sombre
for b = 1:numel(R)
    S = R(b).S;
    subplot(3, 1, 1); hold on; plot(S.t, S.x, 'DisplayName', sprintf('x drone (g=%g)', R(b).g));
    subplot(3, 1, 2); hold on; plot(S.t, S.z, 'DisplayName', sprintf('z (g=%g)', R(b).g));
    subplot(3, 1, 3); hold on; plot(S.t, S.psi, 'DisplayName', sprintf('\\psi (g=%g)', R(b).g));
end
S = R(1).S;
subplot(3, 1, 1); plot(S.t, S.x_v, 'k--', 'LineWidth', 1.5, 'DisplayName', 'voiture'); plot(S.t, S.x_ref, 'k:', 'DisplayName', 'x_{ref}');
ylabel('x (m)'); legend('Location', 'northwest'); grid on;
title(sprintf('Mission : decollage 0.5 m, scan 360°, poursuite, posé sur la voiture (v = %.1f m/s)', v_v));
subplot(3, 1, 2); yyaxis right; plot(S.t, S.phase, 'DisplayName', 'phase'); ylabel('phase'); yyaxis left; ylabel('z (m)'); legend; grid on;
subplot(3, 1, 3); yline(2*pi, 'k:', '2\pi', 'HandleVisibility', 'off'); ylabel('\psi (rad)'); xlabel('t (s)'); legend('Location', 'southeast'); grid on;
exportgraphics(f, fullfile(dossier, 'mission_LQ.png'));
close_system(mdl, 0);
