%% VÉRIFICATION DE L'INSTALLATION (MATLAB) : tout est-il à sa place et cohérent avec Python ?
%  À quoi sert ce script : après un clonage du dépôt ou un rangement, vérifier en quelques minutes, SANS écraser
%  aucun résultat (aucun vol bruité n'est relancé) :
%    1. commande LQ : gains_LQI.m s'exécute, les 4 modèles test_LQI_*.slx et mission_LQ.slx simulent 1 s ;
%       gains_LQ_ARX_attitude.m relit bien le modèle ARX identifié (python/modeles/attitude/) ;
%    2. RL : pour chaque étape (z, zpsi, x, xy, mission), le contrôleur MATLAB rejoue le vol Python exporté pas à pas
%       (mêmes observations, actions, commandes), le modèle EINT_RL_closed_loop_<étape>.slx est (re)construit et simule 2 s.
%  Lancer depuis n'importe où :  run('<chemin>/EI_RL/verifier_installation.m')
%  Prérequis : les exports Python (python/export/*.py) ont été faits (ils le sont dans le dépôt).

%% 1. Commande LQ (LQI double intégrateur)
EI_RACINE = fileparts(mfilename('fullpath'));
run(fullfile(EI_RACINE, 'commande_LQ', 'gains_LQI.m'));          % contient clear : on redéfinit les chemins après
EI_RACINE = fileparts(which('verifier_installation'));
if isempty(EI_RACINE), EI_RACINE = pwd; end
LQ = fullfile(EI_RACINE, 'commande_LQ'); addpath(LQ);
for nom = {'test_LQI_z', 'test_LQI_psi', 'test_LQI_x', 'test_LQI_y'}
    load_system(fullfile(LQ, [nom{1} '.slx']));
    sim(nom{1}, 'StopTime', '1');
    close_system(nom{1}, 0);
    fprintf('OK  commande_LQ/%s.slx simule\n', nom{1});
end
run(fullfile(LQ, 'mission_LQ', 'parametres_mission_LQ.m'));          % gains + voiture (contient clear)
EI_RACINE = fileparts(which('verifier_installation')); LQ = fullfile(EI_RACINE, 'commande_LQ');
load_system(fullfile(LQ, 'mission_LQ', 'mission_LQ.slx')); sim('mission_LQ', 'StopTime', '1'); close_system('mission_LQ', 0);
fprintf('OK  commande_LQ/mission_LQ/mission_LQ.slx simule\n');
run(fullfile(LQ, 'LQ_ARX_attitude', 'gains_LQ_ARX_attitude.m'));    % relit le modèle ARX, recalcule K (déterministe)
close all;
fprintf('OK  commande_LQ/LQ_ARX_attitude/gains_LQ_ARX_attitude.m\n');

%% 2. RL : vérification pas à pas contre Python, reconstruction et court vol de chaque boucle
EI_RACINE = fileparts(which('verifier_installation'));
SRL = fullfile(EI_RACINE, 'simulink_RL');
addpath(fullfile(EI_RACINE, 'Matlab files'), fullfile(SRL, 'z'));
verifier_z(fullfile(EI_RACINE, 'rl_interface'));
fprintf('OK  étape z (rl_interface/rl_controller.m = Python)\n');
for etape = {'zpsi', 'x', 'xy', 'mission'}
    e = etape{1}; ici = fullfile(SRL, e); addpath(ici); cd(ici);
    feval(['verifier_' e], ici);
    feval(['construire_boucle_' e]);                                    % (re)crée EINT_RL_closed_loop_<e>.slx
    random_init; Ts = 0.05; T_end = 2; g_noise = 0;
    clear(['rl_controller_' e]); clear controleur_zpsi controleur_x controleur_xy
    out = sim(['EINT_RL_closed_loop_' e]);
    L = out.(['rl_log_' e]);
    fprintf('OK  étape %-8s : vérification, modèle reconstruit, vol de 2 s (z final %.3f m)\n', e, L(end, 4));
    clear(['rl_controller_' e]); rmpath(ici);
end
cd(EI_RACINE);
fprintf('\nINSTALLATION VÉRIFIÉE : commande LQ et 5 étapes RL.\n');
