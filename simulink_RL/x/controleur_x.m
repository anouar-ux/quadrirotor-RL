function [u_z, a_psi, a_x, refs, obs_x] = controleur_x(t, z, vz, az_prev, u_z_prev, psi, wz, wdz_prev, a_psi_prev, ...
                                                       x, vx, theta, wy, a_x_prev)
%CONTROLEUR_X  Les trois politiques (z, lacet, x) pour un pas, à partir des mesures.
%   Même calcul que env_z.py (z), env_psi.py (lacet) et env_x.py (x, mission MISSION_X de export_x.py).
%   refs = [z_ref ; psi_ref ; x_ref ; vx_ref]. a_x donne theta_ref = THETA_MAX * a_x (suivi par le LQ d'attitude,
%   dans rl_controller_x.m). Mémoire (intégrales) en persistent : "clear controleur_x" au début de chaque vol.
persistent PZ PPSI PX int_z int_psi int_x
if isempty(PZ)
    ici = fileparts(mfilename('fullpath'));
    PZ = charger(fullfile(ici, 'policy_z.mat'));
    PPSI = charger(fullfile(ici, 'policy_psi.mat'));
    PX = charger(fullfile(ici, 'policy_x.mat'));
    int_z = 0; int_psi = 0; int_x = 0;
end
TS = 0.05;
% --- z : observation() de env_z.py (6 composantes)
[z_ref, vz_ref] = consigne_z(t);
e = z_ref - z;
if abs(e) < 0.1, int_z = int_z + e * TS; end
int_z = min(max(int_z, -0.5), 0.5);
u_z = reseau(PZ, [e / 0.5; vz; az_prev / 10; u_z_prev; int_z / 0.5; vz_ref - vz]);
% --- lacet : observation() de env_psi.py (6 composantes) ; consigne : cap tenu à 0
psi_ref = 0; w_ref = 0;
e = psi_ref - psi;
if abs(e) < 0.1, int_psi = int_psi + e * TS; end
int_psi = min(max(int_psi, -0.5), 0.5);
a_psi = reseau(PPSI, [e / 0.5; wz; wdz_prev / 5; a_psi_prev; int_psi / 0.5; w_ref - wz]);
% --- x : observation() de env_x.py (7 composantes)
[x_ref, vx_ref] = consigne_x(t);
e = x_ref - x;
if abs(e) < 0.1, int_x = int_x + e * TS; end
int_x = min(max(int_x, -0.5), 0.5);
obs_x = [e / 2; vx; theta / 0.2; wy; a_x_prev; int_x / 0.5; vx_ref - vx];
a_x = reseau(PX, obs_x);
refs = [z_ref; psi_ref; x_ref; vx_ref];
end

function [z_ref, v_ref] = consigne_z(t)
% Décollage à 0,5 m à t = 1 s (sauts de hauteur lissés, 2e ordre critique, TAU = 0,35 s, comme env_z.py)
TS = 0.05; TAU = 0.35;
PALIERS = [0.0, 0.0;
           1.0, 0.5];
t = round(t / TS) * TS;
z_ref = PALIERS(1, 2); v_ref = 0;
for i = 2:size(PALIERS, 1)
    if t >= PALIERS(i, 1)
        s = t - PALIERS(i, 1); dh = PALIERS(i, 2) - PALIERS(i - 1, 2);
        z_ref = z_ref + dh * (1 - (1 + s / TAU) * exp(-s / TAU));
        v_ref = v_ref + dh * s / TAU^2 * exp(-s / TAU);
    end
end
end

function [x_ref, v_ref] = consigne_x(t)
% MISSION_X de export_x.py : consigne_x() de env_x.py, sauts de POSITION (type 1) et de VITESSE (type 2),
% lissés par un 2e ordre critique de constante TAU_X = 1 s.
TS = 0.05; TAU = 1.0;
EV = [ 4.0, 1,  2.0;        % saut de +2 m
      12.0, 1, -2.0;        % retour
      20.0, 1,  5.5;        % la voiture apparaît 5,5 m devant...
      20.0, 2,  0.3];       % ... et roule à 0,3 m/s
t = round(t / TS) * TS;
x_ref = 0; v_ref = 0; v_prec = 0;
for i = 1:size(EV, 1)
    if t >= EV(i, 1)
        s = t - EV(i, 1); e = exp(-s / TAU);
        r = 1 - (1 + s / TAU) * e; dr = s / TAU^2 * e;
        if EV(i, 2) == 1
            x_ref = x_ref + EV(i, 3) * r; v_ref = v_ref + EV(i, 3) * dr;
        else
            dv = EV(i, 3) - v_prec;
            x_ref = x_ref + dv * (s - 2 * TAU + (s + 2 * TAU) * e); v_ref = v_ref + dv * r;
        end
    end
    if EV(i, 2) == 2, v_prec = EV(i, 3); end
end
end

function P = charger(fichier)
P = load(fichier);
P.acts = strsplit(P.activations, ',');
end

function u = reseau(P, obs)
% Même calcul que rl_controller.m (fourni) : couches, activations, sortie bornée (PPO : 'clip')
x = obs(:);
for i = 1:double(P.n_layers)
    x = P.(sprintf('W%d', i)) * x + P.(sprintf('b%d', i));
    switch P.acts{i}
        case 'relu', x = max(x, 0);
        case 'tanh', x = tanh(x);
    end
end
u = min(max(x(1), P.action_low(1)), P.action_high(1));
end
