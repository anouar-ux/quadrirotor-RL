function [u_z, a_psi, z_ref, psi_ref, obs_z, obs_psi] = controleur_zpsi(t, z, vz, az_prev, u_z_prev, psi, wz, wdz_prev, a_psi_prev)
%CONTROLEUR_AVANCE  Les deux politiques (z et lacet) pour un pas, à partir des mesures.
%   Même calcul que env_z.py (z, mission MISSION_AVANCEE_Z) et env_psi.py (lacet, MISSION_SCAN).
%   Mémoire (intégrales) dans des variables persistent : "clear controleur_zpsi" au début de chaque vol.
persistent PZ PPSI int_z int_psi
if isempty(PZ)
    ici = fileparts(mfilename('fullpath'));
    PZ = charger(fullfile(ici, 'policy_z.mat'));
    PPSI = charger(fullfile(ici, 'policy_psi.mat'));
    int_z = 0; int_psi = 0;
end
TS = 0.05;
% --- z : mêmes 6 composantes que observation() de env_z.py
[z_ref, v_ref] = consigne_z(t);
e = z_ref - z;
if abs(e) < 0.1, int_z = int_z + e * TS; end
int_z = min(max(int_z, -0.5), 0.5);
obs_z = [e / 0.5; vz; az_prev / 10; u_z_prev; int_z / 0.5; v_ref - vz];
u_z = reseau(PZ, obs_z);
% --- lacet : mêmes 6 composantes que observation() de env_psi.py
[psi_ref, w_ref] = consigne_psi(t);
e = psi_ref - psi;
if abs(e) < 0.1, int_psi = int_psi + e * TS; end
int_psi = min(max(int_psi, -0.5), 0.5);
obs_psi = [e / 0.5; wz; wdz_prev / 5; a_psi_prev; int_psi / 0.5; w_ref - wz];
a_psi = reseau(PPSI, obs_psi);
end

function [z_ref, v_ref] = consigne_z(t)
% MISSION_AVANCEE_Z de env_z.py : sauts de hauteur lissés (2e ordre critique, TAU = 0.35 s)
TS = 0.05; TAU = 0.35;
PALIERS = [0.0,   0.0;         % au départ
           2.0,   0.5;         % montée à 0.5 m
          15.0, -0.25];        % atterrissage au sol
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

function [psi_ref, w_ref] = consigne_psi(t)
% MISSION_SCAN de env_psi.py : sauts de VITESSE de lacet lissés ; un tour complet entre 5 et 13 s
TS = 0.05; TAU = 0.35;
W = 2 * pi / (13.0 - 5.0);
SAUTS = [5.0, W;               % début du scan : vitesse W
        13.0, 0.0];            % fin du scan : arrêt
t = round(t / TS) * TS;
psi_ref = 0; w_ref = 0; w_prec = 0;
for i = 1:size(SAUTS, 1)
    if t >= SAUTS(i, 1)
        x = t - SAUTS(i, 1); dw = SAUTS(i, 2) - w_prec;
        psi_ref = psi_ref + dw * (x - 2 * TAU + (x + 2 * TAU) * exp(-x / TAU));
        w_ref = w_ref + dw * (1 - (1 + x / TAU) * exp(-x / TAU));
    end
    w_prec = SAUTS(i, 2);
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
