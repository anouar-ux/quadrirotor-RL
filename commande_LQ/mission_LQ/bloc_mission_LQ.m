function [x_ref, vx_ref, y_ref, z_ref, psi_ref, phase, x_v] = bloc_mission_LQ(t, x, vx, z, x_v0, v_v, h_v)
% BLOC_MISSION_LQ  Code du bloc « mission » de mission_LQ.slx, POUR LECTURE (le modèle contient sa propre copie).
% Séquenceur de mission : décollage -> scan 360° -> poursuite -> descente -> posé
% Appelé toutes les 0.05 s (pas d'échantillonnage du bloc).
persistent ph d0 t3 t_ok t4
if isempty(ph), ph = 1; d0 = 0; t3 = 0; t_ok = 0; t4 = 0; end

Z_VOL    = 0.5;   % altitude de vol (m)
T_MONTEE = 5;     % durée de la rampe de décollage (s)
T_SCAN   = 10;    % durée du tour complet (s) -> 0.63 rad/s
T_APP    = 3;     % constante de temps de l'approche (s)
V_DESC   = 0.1;   % vitesse de descente (m/s)
TOL_X    = 0.05;  % tolérance d'alignement en position (m)
TOL_V    = 0.05;  % tolérance d'alignement en vitesse (m/s)

x_v = x_v0 + v_v * t;            % voiture : mouvement rectiligne uniforme selon x
x_ref = 0; vx_ref = 0; y_ref = 0;
z_ref = Z_VOL * min(t / T_MONTEE, 1);
psi_ref = 0;

% --- transitions et consignes
if ph == 1 && t >= T_MONTEE, ph = 2; end
if ph >= 2
    psi_ref = 2 * pi * min((t - T_MONTEE) / T_SCAN, 1);   % vitesse constante puis maintien à 2*pi
end
if ph == 2 && t >= T_MONTEE + T_SCAN
    ph = 3; t3 = t; d0 = x_v - x;                          % écart au début de la poursuite
end
if ph >= 3
    s = (t - t3) / T_APP;
    e = exp(-s);
    x_ref  = x_v - d0 * (1 + s) * e;                       % approche critique : part du drone, vitesse relative initiale nulle
    vx_ref = v_v + d0 / T_APP * s * e;
    z_ref  = Z_VOL;
end
if ph == 3
    if abs(x - x_v) < TOL_X && abs(vx - v_v) < TOL_V, t_ok = t_ok + 0.05; else, t_ok = 0; end
    if t_ok >= 1, ph = 4; t4 = t; end
end
if ph >= 4
    z_ref = max(Z_VOL - V_DESC * (t - t4), h_v);
    if ph == 4 && z - h_v < 0.02, ph = 5; end
end
phase = ph;
end
