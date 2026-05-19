%% VALIDATE_FRUIT_SORTING - Offline simulation of the complete sorting pipeline
% Tests FK, IK, trajectory generation, and sorting controller WITHOUT hardware.
% Generates plots showing the robot motion for the report.

clear; clc; close all;
addpath('../matlab_code');

fprintf('=== Fruit Sorting Validation ===\n\n');

%% ---- Test 1: FK/IK Round-trip Validation ----
fprintf('--- Test 1: FK/IK Round-trip ---\n');
test_positions = [
    0.25,  0.25,  0.10;   % Front-right
    0.30, -0.20,  0.05;   % Strawberry basket
   -0.30, -0.20,  0.05;   % Banana basket
    0.00, -0.35,  0.05;   % Tomato basket
    0.00,  0.375, 0.10;   % Default Lab 5 position
    0.45,  0.00,  0.49;   % Home
]';

fk_ik_errors = zeros(1, size(test_positions, 2));
for i = 1:size(test_positions, 2)
    p_target = test_positions(:, i);
    [phi_opt, ~] = qarm_IK(p_target, 0);
    [p_result, ~] = qarm_FK(phi_opt);
    fk_ik_errors(i) = norm(p_result - p_target) * 1000;  % mm
    fprintf('  Target: [%.3f, %.3f, %.3f] -> Error: %.3f mm', ...
        p_target(1), p_target(2), p_target(3), fk_ik_errors(i));
    if fk_ik_errors(i) < 1.0
        fprintf(' [PASS]\n');
    else
        fprintf(' [CHECK]\n');
    end
end
fprintf('  Max FK/IK round-trip error: %.3f mm\n\n', max(fk_ik_errors));

%% ---- Test 2: Trajectory Smoothness ----
fprintf('--- Test 2: Cubic Trajectory Smoothness ---\n');
p_start = [0.45; 0; 0.49];
p_end   = [0.25; 0.25; 0.10];
T = 2.0;
dt = 0.002;
t_vec = 0:dt:T;
pos_traj = zeros(3, length(t_vec));
vel_traj = zeros(3, length(t_vec));

for k = 1:length(t_vec)
    pos_traj(:, k) = cubicTrajectory(p_start, p_end, T, t_vec(k));
end

% Numerical velocity
for k = 2:length(t_vec)
    vel_traj(:, k) = (pos_traj(:, k) - pos_traj(:, k-1)) / dt;
end

max_vel = max(abs(vel_traj), [], 2);
fprintf('  Max velocity: vx=%.3f, vy=%.3f, vz=%.3f m/s\n', max_vel);
fprintf('  Start/end velocity check: v_start=%.6f, v_end=%.6f m/s [should be ~0]\n', ...
    norm(vel_traj(:, 2)), norm(vel_traj(:, end)));

%% ---- Test 3: Full Sorting Simulation ----
fprintf('\n--- Test 3: Full Sorting Simulation ---\n');

% Simulated fruit positions in workspace
fruit_positions = [
    0.20,  0.30,  0.02;   % Strawberry 1
    0.15,  0.25,  0.02;   % Tomato 1
   -0.10,  0.35,  0.02;   % Banana 1
    0.25,  0.20,  0.02;   % Strawberry 2
    0.00,  0.30,  0.02;   % Banana 2
   -0.15,  0.25,  0.02;   % Tomato 2
]';

fruit_types = {'strawberry'; 'tomato'; 'banana'; 'strawberry'; 'banana'; 'tomato'};

% Simulate controller
dt = 0.002;
T_sim = 120;  % 2 minutes max
t_sim = 0:dt:T_sim;
n_steps = length(t_sim);

ee_positions = zeros(3, n_steps);
joint_angles = zeros(4, n_steps);
gripper_log = zeros(1, n_steps);
state_log = cell(1, n_steps);

% Initial position (home)
current_joints = [0; 0; 0; 0];
[current_ee, ~] = qarm_FK(current_joints);

operator_cmd = struct('vx', 0, 'vy', 0, 'vz', 0, 'gripper', 0);

fprintf('  Simulating %d seconds at 500Hz...\n', T_sim);

done = false;
for k = 1:n_steps
    [joint_cmd, grip_cmd, state_str, status] = fruitSortingController( ...
        current_joints, current_ee, fruit_positions, fruit_types, ...
        'auto', operator_cmd, t_sim(k));

    % Record
    joint_angles(:, k) = joint_cmd;
    [ee_pos, ~] = qarm_FK(joint_cmd);
    ee_positions(:, k) = ee_pos;
    gripper_log(k) = grip_cmd;
    state_log{k} = state_str;

    current_joints = joint_cmd;
    current_ee = ee_pos;

    if strcmp(state_str, 'DONE') && ~done
        fprintf('  DONE at t=%.1fs: %s\n', t_sim(k), status);
        done = true;
        % Continue a bit more to show steady state
        if k + 500 <= n_steps
            % Trim simulation
            n_steps = k + 500;
        end
        break;
    end
end

% Trim to actual length
ee_positions = ee_positions(:, 1:min(k+500, size(ee_positions,2)));
joint_angles = joint_angles(:, 1:min(k+500, size(joint_angles,2)));
gripper_log = gripper_log(1:min(k+500, length(gripper_log)));
t_plot = t_sim(1:size(ee_positions,2));

%% ---- Generate Report Figures ----
fprintf('\n--- Generating Figures ---\n');

% Figure 1: 3D Trajectory
figure('Name', 'Fruit Sorting - 3D Trajectory', 'Position', [100 100 800 600]);
plot3(ee_positions(1,:), ee_positions(2,:), ee_positions(3,:), 'b-', 'LineWidth', 1);
hold on;
% Plot fruit positions
colors = containers.Map({'strawberry','tomato','banana'}, {'r','m','y'});
for i = 1:length(fruit_types)
    c = colors(fruit_types{i});
    plot3(fruit_positions(1,i), fruit_positions(2,i), fruit_positions(3,i), ...
        'o', 'MarkerSize', 12, 'MarkerFaceColor', c, 'MarkerEdgeColor', 'k');
end
% Plot baskets
plot3(0.30, -0.20, 0.05, 's', 'MarkerSize', 15, 'MarkerFaceColor', 'r', 'DisplayName', 'Strawberry Basket');
plot3(-0.30, -0.20, 0.05, 's', 'MarkerSize', 15, 'MarkerFaceColor', 'y', 'DisplayName', 'Banana Basket');
plot3(0.00, -0.35, 0.05, 's', 'MarkerSize', 15, 'MarkerFaceColor', 'm', 'DisplayName', 'Tomato Basket');
% Plot home
plot3(0.45, 0, 0.49, 'kp', 'MarkerSize', 15, 'MarkerFaceColor', 'g', 'DisplayName', 'Home');
xlabel('X (m)'); ylabel('Y (m)'); zlabel('Z (m)');
title('QArm Fruit Sorting - End-Effector Trajectory');
grid on; view(45, 30);
legend('Trajectory', 'Location', 'best');
saveas(gcf, '../figures/trajectory_3D.png');

% Figure 2: End-Effector Position vs Time
figure('Name', 'EE Position vs Time', 'Position', [100 100 900 500]);
subplot(3,1,1);
plot(t_plot, ee_positions(1,:), 'r-'); ylabel('X (m)'); title('End-Effector Position');
grid on;
subplot(3,1,2);
plot(t_plot, ee_positions(2,:), 'g-'); ylabel('Y (m)');
grid on;
subplot(3,1,3);
plot(t_plot, ee_positions(3,:), 'b-'); ylabel('Z (m)'); xlabel('Time (s)');
grid on;
saveas(gcf, '../figures/ee_position_time.png');

% Figure 3: Joint Angles vs Time
figure('Name', 'Joint Angles', 'Position', [100 100 900 600]);
labels = {'phi_1', 'phi_2', 'phi_3', 'phi_4'};
for j = 1:4
    subplot(4,1,j);
    plot(t_plot, rad2deg(joint_angles(j,1:length(t_plot))), 'b-');
    ylabel([labels{j} ' (deg)']);
    grid on;
    if j == 1, title('Joint Angles During Sorting'); end
    if j == 4, xlabel('Time (s)'); end
end
saveas(gcf, '../figures/joint_angles_time.png');

% Figure 4: Gripper State
figure('Name', 'Gripper State', 'Position', [100 100 900 200]);
stairs(t_plot, gripper_log(1:length(t_plot)), 'k-', 'LineWidth', 2);
yticks([0 1]); yticklabels({'Open', 'Closed'});
xlabel('Time (s)'); title('Gripper State During Sorting');
grid on;
saveas(gcf, '../figures/gripper_state.png');

% Figure 5: Top View (XY plane)
figure('Name', 'Top View', 'Position', [100 100 600 600]);
plot(ee_positions(1,:), ee_positions(2,:), 'b-', 'LineWidth', 0.5);
hold on;
for i = 1:length(fruit_types)
    c = colors(fruit_types{i});
    plot(fruit_positions(1,i), fruit_positions(2,i), ...
        'o', 'MarkerSize', 12, 'MarkerFaceColor', c, 'MarkerEdgeColor', 'k');
end
plot(0.30, -0.20, 's', 'MarkerSize', 15, 'MarkerFaceColor', 'r');
plot(-0.30, -0.20, 's', 'MarkerSize', 15, 'MarkerFaceColor', 'y');
plot(0.00, -0.35, 's', 'MarkerSize', 15, 'MarkerFaceColor', 'm');
xlabel('X (m)'); ylabel('Y (m)');
title('Top View - Sorting Workspace');
axis equal; grid on;
saveas(gcf, '../figures/top_view.png');

fprintf('\nAll figures saved to ../figures/\n');
fprintf('=== Validation Complete ===\n');
