function [u_z, a_psi, a_av, a_co, refs, o_av, o_co] = controleur_xy(t, z, vz, az_prev, u_z_prev, psi, wz, wdz_prev, ...
                                                                  a_psi_prev, x, y, vx, vy, theta, phi, wx, wy, mode)
%CONTROLEUR_XY  Étape C3 : politiques z, lacet et position (x, y) pour un pas, à partir des mesures.
%   Même calcul que env_z.py (z), env_psi.py (lacet, scan 360° entre 5 et 13 s) et sim_xy.py (x, y).
%   Position : erreurs, vitesses et vitesses de consigne sont TOURNÉES DE -psi (repère du sol -> repère du drone),
%   puis le MÊME agent x pilote les deux axes du drone :
%       axe avant : obs = [e_av/2, v_av, theta/0.2,  wy, a_av_prec, int_av/0.5, vref_av - v_av] -> theta_ref =  0,2 a_av
%       axe côté  : obs = [e_co/2, v_co, -phi/0.2, -wx, a_co_prec, int_co/0.5, vref_co - v_co] -> phi_ref   = -0,2 a_co
%   (ay = -9,91 phi : avec theta' = -phi, l'axe côté est un axe x identique).
%   mode = 'rl' (agent x) ou 'pd' (référence PD avec anticipation, Kp 0,3, Kd 0,45, mêmes observations).
%   refs = [z_ref ; psi_ref ; x_ref ; y_ref ; vx_ref ; vy_ref].
%   Mémoire (intégrales, actions précédentes) en persistent : "clear controleur_xy" au début de chaque vol.
persistent PZ PPSI PX int_z int_psi int_av int_co a_av_prec a_co_prec
if isempty(PZ)
    ici = fileparts(mfilename('fullpath'));
    PZ = charger(fullfile(ici, 'policy_z.mat'));
    PPSI = charger(fullfile(ici, 'policy_psi.mat'));
    PX = charger(fullfile(ici, 'policy_x.mat'));
    int_z = 0; int_psi = 0; int_av = 0; int_co = 0; a_av_prec = 0; a_co_prec = 0;
end
if nargin < 18, mode = 'rl'; end
TS = 0.05; THETA_MAX = 0.2; I_MAX = 0.5; E_INT = 0.1;
% --- z : observation() de env_z.py
[z_ref, vz_ref] = consigne_z(t);
e = z_ref - z;
if abs(e) < 0.1, int_z = int_z + e * TS; end
int_z = min(max(int_z, -0.5), 0.5);
u_z = reseau(PZ, [e / 0.5; vz; az_prev / 10; u_z_prev; int_z / 0.5; vz_ref - vz]);
% --- lacet : observation() de env_psi.py, consigne MISSION_SCAN
[psi_ref, w_ref] = consigne_psi(t);
e = psi_ref - psi;
if abs(e) < 0.1, int_psi = int_psi + e * TS; end
int_psi = min(max(int_psi, -0.5), 0.5);
a_psi = reseau(PPSI, [e / 0.5; wz; wdz_prev / 5; a_psi_prev; int_psi / 0.5; w_ref - wz]);
% --- position : consignes dans le repère du sol, puis rotation de -psi (psi MESURÉ)
EV_X = [16.0, 1, 2.0;          % diagonale vers (2 ; 1,5)
        24.0, 1, 3.5;          % la voiture apparaît en x = 5,5 m...
        24.0, 2, 0.3];         % ... et roule selon x à 0,3 m/s
EV_Y = [16.0, 1, 1.5;
        24.0, 1, -1.5];        % ... sur l'axe y = 0
[x_ref, vx_ref] = consigne_ev(t, EV_X);
[y_ref, vy_ref] = consigne_ev(t, EV_Y);
c = cos(psi); s = sin(psi);
e_av = c * (x_ref - x) + s * (y_ref - y);   e_co = -s * (x_ref - x) + c * (y_ref - y);
v_av = c * vx + s * vy;                     v_co = -s * vx + c * vy;
vr_av = c * vx_ref + s * vy_ref;            vr_co = -s * vx_ref + c * vy_ref;
if abs(e_av) < E_INT, int_av = int_av + e_av * TS; end
if abs(e_co) < E_INT, int_co = int_co + e_co * TS; end
int_av = min(max(int_av, -I_MAX), I_MAX); int_co = min(max(int_co, -I_MAX), I_MAX);
o_av = [e_av / 2; v_av; theta / THETA_MAX; wy; a_av_prec; int_av / I_MAX; vr_av - v_av];
o_co = [e_co / 2; v_co; -phi / THETA_MAX; -wx; a_co_prec; int_co / I_MAX; vr_co - v_co];
if strcmp(mode, 'pd')
    a_av = pd(o_av); a_co = pd(o_co);
else
    a_av = reseau(PX, o_av); a_co = reseau(PX, o_co);
end
a_av_prec = a_av; a_co_prec = a_co;
refs = [z_ref; psi_ref; x_ref; y_ref; vx_ref; vy_ref];
end

function a = pd(o)
% PD avec anticipation (C2.2) écrit sur l'observation : e = 2 o(1), erreur de vitesse = o(7)
KP = 0.3; KD = 0.45; THETA_MAX = 0.2;
a = min(max(KP * 2 * o(1) + KD * o(7), -THETA_MAX), THETA_MAX) / THETA_MAX;
end

function [z_ref, v_ref] = consigne_z(t)
% Décollage à 0,5 m à t = 1 s (2e ordre critique, TAU = 0,35 s, comme env_z.py)
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

function [psi_ref, w_ref] = consigne_psi(t)
% MISSION_SCAN de env_psi.py : un tour complet entre 5 et 13 s (sauts de vitesse lissés, TAU = 0,35 s)
TS = 0.05; TAU = 0.35;
W = 2 * pi / (13.0 - 5.0);
SAUTS = [5.0, W;
        13.0, 0.0];
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

function [p_ref, v_ref] = consigne_ev(t, EV)
% consigne_x() de env_x.py : sauts de POSITION (type 1) et de VITESSE (type 2), 2e ordre critique TAU_X = 1 s
TS = 0.05; TAU = 1.0;
t = round(t / TS) * TS;
p_ref = 0; v_ref = 0; v_prec = 0;
for i = 1:size(EV, 1)
    if t >= EV(i, 1)
        s = t - EV(i, 1); e = exp(-s / TAU);
        r = 1 - (1 + s / TAU) * e; dr = s / TAU^2 * e;
        if EV(i, 2) == 1
            p_ref = p_ref + EV(i, 3) * r; v_ref = v_ref + EV(i, 3) * dr;
        else
            dv = EV(i, 3) - v_prec;
            p_ref = p_ref + dv * (s - 2 * TAU + (s + 2 * TAU) * e); v_ref = v_ref + dv * r;
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
