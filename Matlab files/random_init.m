%% Initialize random number stream
s = RandStream('mt19937ar','Seed','shuffle');
RandStream.setGlobalStream(s);

Ts = 0.05; % Maybe not the right value...

g_noise = 0.01; % Global noise gain

% Generate random seeds for each state variable to ensure variability in simulations for data generation

seed_x = randi(s, 10000);      % Random seed for x position
seed_y = randi(s, 10000);      % Random seed for y position
seed_z = randi(s, 10000);      % Random seed for z position
seed_vx = randi(s, 10000);     % Random seed for x velocity
seed_vy = randi(s, 10000);     % Random seed for y velocity
seed_vz = randi(s, 10000);     % Random seed for z velocity
seed_ax = randi(s, 10000);     % Random seed for x acceleration
seed_ay = randi(s, 10000);     % Random seed for y acceleration
seed_az = randi(s, 10000);     % Random seed for z acceleration
seed_phi = randi(s, 10000);    % Random seed for roll angle (phi)
seed_psi = randi(s, 10000);    % Random seed for yaw angle (psi)
seed_theta = randi(s, 10000);  % Random seed for pitch angle (theta)
seed_wx = randi(s, 10000);     % Random seed for x angular velocity
seed_wy = randi(s, 10000);     % Random seed for y angular velocity
seed_wz = randi(s, 10000);     % Random seed for z angular velocity
seed_w_dot_x = randi(s, 10000);% Random seed for x angular acceleration
seed_w_dot_y = randi(s, 10000);% Random seed for y angular acceleration
seed_w_dot_z = randi(s, 10000);% Random seed for z angular acceleration