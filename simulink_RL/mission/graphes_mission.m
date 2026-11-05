function graphes_mission(R, P, VO, ici)
%GRAPHES_MISSION  Écart Simulink / Python et figure de la mission finale, à partir des vols enregistrés.
%   Seul : S = load('resultats_mission.mat'); graphes_mission(S.R, S.P, load('voiture.mat'), pwd)
%   (Simulink enregistre N+1 instants, 0 à T_end inclus ; Python N : on compare les N premiers.)
n = size(P.vrai, 1);
noms = {'stabilisation', 'montée', 'scan', 'poursuite', 'descente', 'posé'};
fprintf('Écart Simulink / Python (vol rl sans bruit), position horizontale : max %.3f m, au moment du posé %.3f m\n', ...
    max(hypot(R.rl_g000(1:n, 2) - P.vrai(1:n, 1), R.rl_g000(1:n, 3) - P.vrai(1:n, 2))), ...
    hypot(R.rl_g000(find(R.rl_g000(:, 24) >= 5, 1), 2) - P.vrai(find(P.y(:, 5) >= 5, 1), 1), ...
          R.rl_g000(find(R.rl_g000(:, 24) >= 5, 1), 3) - P.vrai(find(P.y(:, 5) >= 5, 1), 2)));

% ---- figure
A0 = R.rl_g000; A1 = R.rl_g100; D1 = R.pd_g100; t = A0(:, 1);
figure('Position', [40 40 1300 900], 'Name', 'Mission finale sur le simulateur');
theme(gcf, 'light');              % figures en mode clair, même si Windows / MATLAB sont en mode sombre
subplot(2, 2, 1); plot(VO.x(1:1201), VO.y(1:1201), 'Color', [.75 .75 .75], 'LineWidth', 5); hold on;
plot(P.vrai(1:n, 1), P.vrai(1:n, 2), 'k:', 'LineWidth', 1.2);
plot(A0(:, 2), A0(:, 3), A1(:, 2), A1(:, 3), '--', D1(:, 2), D1(:, 3), '-.', 'LineWidth', 1.1);
for L = {A0, A1}
    for p = 3:5
        j = find(L{1}(:, 24) >= p, 1); mk = {'o', 'v', 'p'};
        if ~isempty(j), plot(L{1}(j, 2), L{1}(j, 3), mk{p - 2}, 'Color', 'k', 'MarkerSize', 8, 'MarkerFaceColor', 'w'); end
    end
end
axis equal; grid on; xlabel('x [m]'); ylabel('y [m]');
legend('voiture', 'Python (prédit)', 'agents, g = 0', 'agents, g = 0,1', 'PD x/y, g = 0,1', 'Location', 'northwest');
title('Vue de dessus (rond : début poursuite, triangle : début descente, étoile : posé)');
subplot(2, 2, 2); plot(t, A0(:, 25), 'k:', t, A0(:, 4), t, A1(:, 4), '--', t, D1(:, 4), '-.', 'LineWidth', 1.1);
ylabel('z [m]'); grid on; yyaxis right; stairs(t, A1(:, 24), 'Color', [.5 .5 .5]); ylim([-0.5 5.5]);
yticks(0:5); yticklabels(noms); legend('z_{ref}', 'agents, g = 0', 'agents, g = 0,1', 'PD x/y, g = 0,1', 'phase (g = 0,1)', ...
    'Location', 'east'); title('Altitude et phases'); xlabel('t [s]');
subplot(2, 2, 3);
semilogy(t, max(hypot(A0(:, 2) - A0(:, 29), A0(:, 3) - A0(:, 30)), 1e-4), ...
         t, max(hypot(A1(:, 2) - A1(:, 29), A1(:, 3) - A1(:, 30)), 1e-4), '--', ...
         t, max(hypot(D1(:, 2) - D1(:, 29), D1(:, 3) - D1(:, 30)), 1e-4), '-.', 'LineWidth', 1); hold on;
yline(0.05, 'k:'); yline(0.10, 'k--'); ylim([1e-3 20]); grid on; xlabel('t [s]'); ylabel('|drone - voiture| [m]');
legend('agents, g = 0', 'agents, g = 0,1', 'PD x/y, g = 0,1', 'alignement 5 cm', 'pause descente 10 cm');
title('Écart horizontal à la voiture');
subplot(2, 2, 4); plot(VO.t(1:numel(t)), unwrap(VO.cap(1:numel(t))) + 2 * pi, 'Color', [.75 .75 .75], 'LineWidth', 4); hold on;
plot(t, A1(:, 26), 'k:', 'LineWidth', 1.2); plot(t, A1(:, 10), 'Color', [0 0.45 0.74]); ylabel('\psi [rad]'); grid on;
yyaxis right; plot(t, A1(:, 9), '-', 'Color', [0.85 0.33 0.10]); hold on; plot(t, A1(:, 8), '-', 'Color', [0.47 0.67 0.19]); ax = gca; ax.YAxis(2).Color = 'k'; ylabel('\theta, \phi [rad]'); xlabel('t [s]');
legend('cap voiture (+2\pi)', '\psi_{ref}', '\psi', '\theta', '\phi', 'Location', 'east'); title('Cap et inclinaisons (agents, g = 0,1)');
exportgraphics(gcf, fullfile(ici, 'mission_simulink_cap.png'), 'Resolution', 120);

end
