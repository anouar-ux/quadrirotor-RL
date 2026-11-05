function u = controleur_essais_interface(v)
% Contrôleur de test (étape A) : tient z = 0.5 m avec le LQI (gains / 2, stables à 50 ms),
% puis ajoute aux 4 moteurs une impulsion différentielle MOTIF * AMP entre T_ON et T_OFF.
global MOTIF AMP T_ON T_OFF
persistent integ
if isempty(integ), integ = 0; end
t = v(1); y = v(2:end);
z = y(3); vz = y(6);
TAU = 0.35; s = round(t / 0.05) * 0.05;
z_ref = 0.5 * (1 - (1 + s / TAU) * exp(-s / TAU));
e = z_ref - z;
integ = integ + e * 0.05;
u_z = 0.38 + 0.93 * e - 0.44 * vz + 0.395 * integ;
d = 0; if t >= T_ON && t < T_OFF, d = AMP; end
u = min(max(u_z + d * MOTIF(:), -1), 1);
end
