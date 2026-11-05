%% PARAMÈTRES DE LA MISSION LQ (atterrissage sur une voiture en mouvement rectiligne uniforme)
%  À quoi sert ce script : préparer l'espace de travail de mission_LQ.slx :
%    1. tous les gains du LQI (z, psi, theta/phi, PD de position) : ../gains_LQI.m ;
%    2. les paramètres de la voiture ci-dessous.
%  Les autres réglages de la mission (altitude 0,5 m, durée du scan, approche, vitesse de descente, tolérances) sont en
%  tête du bloc « mission » du modèle (code lisible dans bloc_mission_LQ.m).
%  Remplace atterissage.m / atterissage_mission.m, dont il reprend exactement les valeurs.
run(fullfile(fileparts(mfilename('fullpath')), '..', 'gains_LQI.m'));     % contient clear : à lancer en premier

% Voiture : x_v(t) = x_v0 + v_v * t, y_v = 0
x_v0 = 1;      % position initiale de la voiture [m]
v_v  = 0.3;    % vitesse de la voiture [m/s]
h_v  = 0;      % hauteur du toit de la voiture = altitude de posé [m]
fprintf('Mission : voiture en x_v0 = %g m, v_v = %g m/s, h_v = %g m\n', x_v0, v_v, h_v);
