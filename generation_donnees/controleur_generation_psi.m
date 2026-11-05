function u = controleur_generation_psi(v)
% Contrôleur de génération de données (lacet) : appelé à chaque pas par le modèle construit dans generation_donnees_psi.m.
%   z : LQI (gains / 2, stables à 50 ms) vers 0.5 m -> altitude tenue pendant tout l'essai
%   ψ : u_psi = EXCITATION(k) - K_W * wz  (faible amortissement de la vitesse : sans lui, le lacet sans
%       frottement dériverait sans limite ; l'excitation reste le signal dominant)
% v = [t ; 18 sorties nommées de UAV_model]. u = commandes des 4 moteurs (mélangeur « + » mesuré à l'étape A).
global EXCITATION K_W T_DEBUT
persistent integ
if isempty(integ), integ = 0; end
TS = 0.05; TAU = 0.35;
t = v(1); y = v(2:end);
z = y(3); vz = y(6); wz = y(12);
s = round(t / TS) * TS;
z_ref = 0.5 * (1 - (1 + s / TAU) * exp(-s / TAU));
e = z_ref - z;
integ = integ + e * TS;
u_z = 0.38 + 0.93 * e - 0.44 * vz + 0.395 * integ;
u_psi = 0;
k = round((t - T_DEBUT) / TS) + 1;                % numéro du pas d'excitation (1 au début)
if k >= 1 && k <= numel(EXCITATION)
    u_psi = EXCITATION(k) - K_W * wz;
end
u = min(max(u_z + u_psi * [1; -1; 1; -1], -1), 1);
end
