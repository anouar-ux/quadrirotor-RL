function y = rl_controller_mission(v)
%RL_CONTROLLER_MISSION  Mission finale, appelé à chaque pas par EINT_RL_closed_loop_mission.slx (Interpreted MATLAB Function, Ts).
%   Miroir EXACT de ControleurMission (python/environnements/sim_mission.py).
%   v = [t ; x y z vx vy vz phi theta psi wx wy wz ax ay az wdx wdy wdz]  (18 sorties nommées de UAV_model).
%   y = [u1..u4 ; phase ; z_ref ; psi_ref ; x_ref ; y_ref ; x_voiture ; y_voiture ; theta_ref ; phi_ref ;
%        u_z_agent ; u_theta ; u_phi]  (16 sorties).
%   Phases : 0 stabilisation (2 s) | 1 montée à 0,5 m (-> stable 1 s) | 2 scan 360° en 10 s | 3 poursuite de la voiture
%            (-> alignée à 5 cm / 10 cm/s pendant 2 s) | 4 descente à 0,1 m/s (pause si écart > 10 cm) | 5 posé.
%   Agents : z (env_z), lacet (env_psi), x sur les deux axes du drone après rotation de -psi (sim_xy).
%   Configuration FINALE (CONFIG_FINALE de sim_mission.py) :
%     - cap : pendant la poursuite, psi_ref tourne vers la voiture (au-delà de 2 m), puis prend le cap de la voiture
%       (en deçà de 1 m), à 0,5 rad/s au plus ; la descente attend en plus |psi - cap voiture| < 3° ;
%     - biais : on retranche à l'agent x sa sortie à l'équilibre a(0) (~0,035).
%   (La première version : cap fixe et biais brut, CAP_VOITURE = false et CORR_BIAIS = false.)
%   global MODE_MISSION = 'rl' (défaut) ou 'pd' (référence PD sur x/y). "clear rl_controller_mission" avant chaque vol.
global MODE_MISSION
persistent A0X psi_mem PZ PPSI PX K VOIT phase n_stab n_al k1 k2 k3 D0 z_desc int_z int_psi int_av int_co a_av a_co ...
           az_prev wdz_prev u_z_prev a_psi_prev wdy_prev wdx_prev Uth Uph
TS = 0.05;
T_STAB = 2.0; Z_VOL = 0.5; TAU_Z = 0.35; TOL_Z = 0.02; TOL_VZ = 0.05; N_STAB_Z = 20;
T_SCAN = 10.0; W_SCAN = 2 * pi / T_SCAN; TAU_W = 0.35; TOL_PSI = 0.05; TOL_WZ = 0.1;
TAU_APP = 1.0; TOL_P = 0.05; TOL_V = 0.10; N_ALIGNE = 40;
V_DESC = 0.1; TOL_P_DESC = 0.10; H_V = 0.0; TOL_POSE = 0.02;
THETA_MAX = 0.2; U_PSI_MAX = 0.03; U_EQ = 0.39; U_MAX_ATT = 0.5; I_MAX = 0.5; E_INT = 0.1;
KP_PD = 0.3; KD_PD = 0.45;
CAP_VOITURE = true; CORR_BIAIS = true;
D_LOIN = 2.0; D_PROCHE = 1.0; K_CAP = 1.0; W_CAP_MAX = 0.5; TOL_CAP = 0.05;
if isempty(PZ)
    ici = fileparts(mfilename('fullpath'));
    PZ = charger(fullfile(ici, 'policy_z.mat')); PPSI = charger(fullfile(ici, 'policy_psi.mat'));
    PX = charger(fullfile(ici, 'policy_x.mat'));
    S = load(fullfile(ici, 'lq_attitude.mat')); K = S.K_ATT(:)';
    VOIT = load(fullfile(ici, 'voiture.mat'));
    phase = 0; n_stab = 0; n_al = 0; k1 = -1; k2 = -1; k3 = -1; D0 = [0; 0]; z_desc = Z_VOL;
    int_z = 0; int_psi = 0; int_av = 0; int_co = 0; a_av = 0; a_co = 0;
    az_prev = 0; wdz_prev = 0; u_z_prev = 0; a_psi_prev = 0; wdy_prev = 0; wdx_prev = 0;
    Uth = zeros(5, 1); Uph = zeros(5, 1);
    psi_mem = 0;
    if CORR_BIAIS, A0X = reseau(PX, zeros(7, 1)); else, A0X = 0; end
end
if isempty(MODE_MISSION), mode = 'rl'; else, mode = MODE_MISSION; end
t = v(1); m = v(2:end); k = round(t / TS);
x = m(1); y_ = m(2); z = m(3); vx = m(4); vy = m(5); vz = m(6); phi = m(7); theta = m(8); psi = m(9);
wx = m(10); wy = m(11); wz = m(12); az = m(15); wdx = m(16); wdy = m(17); wdz = m(18);

% ================================================================ superviseur
i = min(k + 1, numel(VOIT.t));
xv = VOIT.x(i); yv = VOIT.y(i); vxv = VOIT.vx(i); vyv = VOIT.vy(i);
capv = VOIT.cap(i); wv = VOIT.v(i) * VOIT.kappa(i);
if phase == 0
    if k * TS >= T_STAB, phase = 1; k1 = k; end
elseif phase == 1
    if abs(z - Z_VOL) < TOL_Z && abs(vz) < TOL_VZ, n_stab = n_stab + 1; else, n_stab = 0; end
    if n_stab >= N_STAB_Z, phase = 2; k2 = k; end
elseif phase == 2
    if (k - k2) * TS >= T_SCAN && abs(psi - 2 * pi) < TOL_PSI && abs(wz) < TOL_WZ
        phase = 3; k3 = k; D0 = [x - xv; y_ - yv];
    end
elseif phase == 3
    ok = hypot(x - xv, y_ - yv) < TOL_P && hypot(vx - vxv, vy - vyv) < TOL_V;
    if CAP_VOITURE, ok = ok && abs(wrapa(psi - capv)) < TOL_CAP; end
    if ok, n_al = n_al + 1; else, n_al = 0; end
    if n_al >= N_ALIGNE, phase = 4; z_desc = Z_VOL; end
elseif phase == 4
    if z - H_V < TOL_POSE, phase = 5; end
end
% consignes
if phase == 0
    z_ref = 0; vz_ref = 0;
elseif phase <= 3
    [r, dr] = lisse((k - k1) * TS, TAU_Z); z_ref = Z_VOL * r; vz_ref = Z_VOL * dr;
elseif phase == 4
    descend = hypot(x - xv, y_ - yv) < TOL_P_DESC && z_desc > H_V;
    if descend, vz_ref = -V_DESC; else, vz_ref = 0; end
    z_ref = z_desc;
    if descend, z_desc = max(z_desc - V_DESC * TS, H_V); end
else
    z_ref = H_V; vz_ref = 0;
end
if phase < 2
    psi_ref = 0; w_ref = 0;
else
    s = (k - k2) * TS;
    r1 = lisse(s, TAU_W);
    psi_ref = W_SCAN * (s - 2 * TAU_W + (s + 2 * TAU_W) * exp(-s / TAU_W)); w_ref = W_SCAN * r1;
    if s >= T_SCAN
        s2 = s - T_SCAN; r2 = lisse(s2, TAU_W);
        psi_ref = psi_ref - W_SCAN * (s2 - 2 * TAU_W + (s2 + 2 * TAU_W) * exp(-s2 / TAU_W));
        w_ref = w_ref - W_SCAN * r2;
    end
    if CAP_VOITURE && phase >= 3
        if k == k3, psi_mem = psi_ref; end                % départ : fin du scan (2 pi)
        d = hypot(xv - x, yv - y_);
        beta = min(max((d - D_PROCHE) / (D_LOIN - D_PROCHE), 0), 1);
        vise = capv + beta * wrapa(atan2(yv - y_, xv - x) - capv);
        w_ref = min(max(K_CAP * wrapa(vise - psi_mem) + (1 - beta) * wv, -W_CAP_MAX), W_CAP_MAX);
        psi_ref = psi_mem;
        psi_mem = psi_mem + w_ref * TS;
    end
end
if phase < 3
    p_ref = [0; 0]; v_ref = [0; 0];
else
    s = (k - k3) * TS; e = exp(-s / TAU_APP);
    p_ref = [xv; yv] + D0 * (1 + s / TAU_APP) * e;
    v_ref = [vxv; vyv] - D0 * s / TAU_APP^2 * e;
end

% ================================================================ agents
e = z_ref - z;  int_z = integre(int_z, e, TS, E_INT, I_MAX);
u_z = reseau(PZ, [e / 0.5; vz; az_prev / 10; u_z_prev; int_z / I_MAX; vz_ref - vz]);
e = psi_ref - psi;  int_psi = integre(int_psi, e, TS, E_INT, I_MAX);
a_psi = reseau(PPSI, [e / 0.5; wz; wdz_prev / 5; a_psi_prev; int_psi / I_MAX; w_ref - wz]);
c = cos(psi); s = sin(psi);
ex = p_ref(1) - x; ey = p_ref(2) - y_;
e_av = c * ex + s * ey;            e_co = -s * ex + c * ey;
v_av = c * vx + s * vy;            v_co = -s * vx + c * vy;
r_av = c * v_ref(1) + s * v_ref(2); r_co = -s * v_ref(1) + c * v_ref(2);
int_av = integre(int_av, e_av, TS, E_INT, I_MAX); int_co = integre(int_co, e_co, TS, E_INT, I_MAX);
o_av = [e_av / 2; v_av; theta / THETA_MAX; wy; a_av; int_av / I_MAX; r_av - v_av];
o_co = [e_co / 2; v_co; -phi / THETA_MAX; -wx; a_co; int_co / I_MAX; r_co - v_co];
if strcmp(mode, 'pd')
    a_av = min(max(KP_PD * 2 * o_av(1) + KD_PD * o_av(7), -THETA_MAX), THETA_MAX) / THETA_MAX;
    a_co = min(max(KP_PD * 2 * o_co(1) + KD_PD * o_co(7), -THETA_MAX), THETA_MAX) / THETA_MAX;
else
    a_av = min(max(reseau(PX, o_av) - A0X, -1), 1); a_co = min(max(reseau(PX, o_co) - A0X, -1), 1);
end
theta_ref = THETA_MAX * a_av; phi_ref = -THETA_MAX * a_co;

% ================================================================ LQ d'attitude, anticipation, mélangeur
u_th = min(max(-K * [theta - theta_ref; wy; wdy; wdy_prev; Uth], -U_MAX_ATT), U_MAX_ATT);
u_ph = min(max(-K * [phi - phi_ref; wx; wdx; wdx_prev; Uph], -U_MAX_ATT), U_MAX_ATT);
u_psi = U_PSI_MAX * a_psi;
u_zc = u_z + U_EQ * (1 / (cos(theta) * cos(phi)) - 1);
u = min(max(u_zc + [u_psi - u_th; -u_psi + u_ph; u_psi + u_th; -u_psi - u_ph], -1), 1);
Uth = [(u(3) - u(1)) / 2; Uth(1:4)]; Uph = [(u(2) - u(4)) / 2; Uph(1:4)];
az_prev = az; wdz_prev = wdz; u_z_prev = u_z; a_psi_prev = a_psi; wdy_prev = wdy; wdx_prev = wdx;
y = [u; phase; z_ref; psi_ref; p_ref; xv; yv; theta_ref; phi_ref; u_z; u_th; u_ph];
end

function a = wrapa(a)
% angle ramené dans [-pi, pi[ (wrap() de sim_mission.py)
a = mod(a + pi, 2 * pi) - pi;
end

function [r, dr] = lisse(s, tau)
e = exp(-s / tau); r = 1 - (1 + s / tau) * e; dr = s / tau^2 * e;
end

function i = integre(i, e, TS, E_INT, I_MAX)
if abs(e) < E_INT, i = i + e * TS; end
i = min(max(i, -I_MAX), I_MAX);
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
