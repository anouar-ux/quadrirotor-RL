function [u, z_ref, obs] = rl_controller(t, z, vz, az_prev, u_prev)
%RL_CONTROLLER  The trained policy, called at each step by EINT_RL_closed_loop.slx.
%   Write mission() and observation() at the end of this file. policy.mat is written by
%   export_for_matlab (export_policy.py).
persistent P
if isempty(P)
    P = load(fullfile(fileparts(mfilename('fullpath')), 'policy.mat'));
    P.acts = strsplit(P.activations, ',');
end
z_ref = mission(t);
m = struct('t', t, 'z', z, 'vz', vz, 'az_prev', az_prev, 'z_ref', z_ref, 'u_prev', u_prev);
obs = double(observation(m));
x = obs(:);
for i = 1:double(P.n_layers)
    x = P.(sprintf('W%d', i)) * x + P.(sprintf('b%d', i));
    switch P.acts{i}
        case 'relu', x = max(x, 0);
        case 'tanh', x = tanh(x);
    end
end
lo = P.action_low(:);
hi = P.action_high(:);
switch P.output
    case 'tanh_scale', x = lo + 0.5 * (tanh(x) + 1) .* (hi - lo);   % SAC
    case 'scale',      x = lo + 0.5 * (x + 1) .* (hi - lo);         % TD3 (tanh in the network)
    otherwise,         x = min(max(x, lo), hi);                     % PPO
end
u = x(1);
end

function obs = observation(m)
% Même calcul et même ordre que observation() + maj_integrale() de drone_env.py.
persistent int_e
if isempty(int_e)
    int_e = 0;                 % remis à 0 par "clear rl_controller" : comme self.int_e = 0 au reset
end
TS = 0.05; I_MAX = 0.5; E_INT = 0.1;

% 1. Intégrale CONDITIONNELLE de l'erreur mesurée (maj_integrale), une fois par appel
e = m.z_ref - m.z;
if abs(e) < E_INT
    int_e = int_e+e*TS;
end
int_e = min(max(int_e, -I_MAX), I_MAX);

% 2. Vitesse de consigne (vitesse_mission), calculée par la même fonction que mission()
[~, v_ref] = consigne(m.t);

% 3. Les 6 composantes, mêmes échelles, même ordre que Python
obs = [ (m.z_ref-m.z)*2 ;      % erreur de suivi
        m.vz ;      % vitesse
        m.az_prev/10 ;      % accélération précédente
        m.u_prev ;      % commande précédente
        int_e/I_MAX ;      % intégrale normalisée
        (v_ref-m.vz) ];     % erreur de vitesse
end

function z_ref = mission(t)
% Consigne lissée de la Figure 1, identique à mission(t, MISSION_FIG1) de drone_env.py.
z_ref = consigne(t);
end

function [z_ref, v_ref] = consigne(t)
% Somme de sauts lissés chacun par un 2e ordre critique (TAU), et sa dérivée exacte.
% PALIERS = MISSION_FIG1 de drone_env.py : [instant du saut [s], hauteur [m]].
TS = 0.05; TAU = 0.35;
PALIERS = [0.0,  0.0;          % sur la plateforme
           3.0,  0.5;          % décollage vers 0.5 m
           7.0, -0.25];        % atterrissage au sol
t = round(t / TS) * TS;        % Simulink peut donner t = 1.9999999 : on revient au pas exact
z_ref = PALIERS(1, 2);
v_ref = 0;
for i = 2:size(PALIERS, 1)
    if t >= PALIERS(i, 1)
        s = t - PALIERS(i, 1);
        dh = PALIERS(i, 2) - PALIERS(i - 1, 2);
        z_ref = z_ref + dh * (1 - (1 + s / TAU) * exp(-s / TAU));
        v_ref = v_ref + dh * s / TAU^2 * exp(-s / TAU);
    end
end
end