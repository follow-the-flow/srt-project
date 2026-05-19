function [joint_cmd, gripper_cmd, state_out, status_msg] = fruitSortingController( ...
    joint_pos, ee_pos, fruit_positions, fruit_types, mode, operator_cmd, t)
%FRUITSORTINGCONTROLLER Main control logic for fruit sorting
%   Implements both AUTONOMOUS and REMOTE modes for the QArm fruit sorting task.
%
%   INPUT:  joint_pos        - 4x1 current joint angles (rad)
%           ee_pos           - 3x1 current end-effector position (m)
%           fruit_positions  - 3xN detected fruit positions in base frame (m)
%           fruit_types      - Nx1 cell array of fruit types
%           mode             - 'auto' or 'remote'
%           operator_cmd     - struct with fields: vx, vy, vz, gripper (for remote)
%           t                - current time (s)
%   OUTPUT: joint_cmd        - 4x1 joint angle commands (rad)
%           gripper_cmd      - scalar: 0 (open) or 1 (closed)
%           state_out        - current state string for display
%           status_msg       - human-readable status message

    %% ---- Persistent State ----
    persistent state fruit_queue current_target target_basket;
    persistent traj_start_time traj_waypoints traj_durations traj_total_time;
    persistent gripper_state sorted_count;

    if isempty(state)
        state = 'INIT';
        fruit_queue = {};
        current_target = [];
        target_basket = [];
        traj_start_time = 0;
        traj_waypoints = [];
        traj_durations = [];
        traj_total_time = 0;
        gripper_state = 0;
        sorted_count = 0;
    end

    %% ---- Basket Positions (must match physical setup) ----
    % Three baskets placed around the QArm workspace
    basket_positions = struct();
    basket_positions.strawberry = [0.30; -0.20; 0.05];   % Right-front
    basket_positions.banana     = [-0.30; -0.20; 0.05];  % Left-front
    basket_positions.tomato     = [0.00; -0.35; 0.05];   % Center-front

    %% ---- Key Positions ----
    home_pos    = [0.45; 0; 0.49];      % QArm home (safe hover)
    safe_z      = 0.20;                  % Safe height for transit
    pick_z      = 0.02;                  % Pick height (near surface)
    place_z     = 0.10;                  % Place/drop height above basket
    approach_z  = 0.15;                  % Approach height above target

    %% ---- Trajectory Timing ----
    T_transit   = 2.0;   % Time for large moves (seconds)
    T_approach  = 1.0;   % Time for approach/retreat
    T_pick      = 0.8;   % Time for final descent to pick
    T_dwell     = 0.5;   % Dwell time for gripper actuation

    %% ===== REMOTE MODE =====
    if strcmp(mode, 'remote')
        state_out = 'REMOTE';
        status_msg = 'Remote control active';

        % Convert operator velocity commands to position increment
        dt = 0.002;  % 500 Hz
        dp = [operator_cmd.vx; operator_cmd.vy; operator_cmd.vz] * dt;
        target_pos = ee_pos + dp;

        % IK to get joint commands
        [joint_cmd, ~] = qarm_IK(target_pos, 0);
        gripper_cmd = operator_cmd.gripper;
        return;
    end

    %% ===== AUTONOMOUS MODE (State Machine) =====
    state_out = state;
    status_msg = state;  % Default status

    switch state
        case 'INIT'
            % Build sorting queue from detected fruits
            if ~isempty(fruit_positions)
                fruit_queue = {};
                for i = 1:size(fruit_positions, 2)
                    entry.pos = fruit_positions(:, i);
                    entry.type = fruit_types{i};
                    fruit_queue{end+1} = entry;
                end
                state = 'GO_HOME';
                status_msg = sprintf('Detected %d fruits. Starting sort.', length(fruit_queue));
            else
                status_msg = 'Waiting for fruit detection...';
                joint_cmd = joint_pos;
                gripper_cmd = 0;
                return;
            end

        case 'GO_HOME'
            % Move to home/safe position
            wp = [ee_pos, home_pos];
            dur = T_transit;
            setupTrajectory(wp, dur, t);
            state = 'MOVING_HOME';
            status_msg = 'Moving to home position';

        case 'MOVING_HOME'
            [pos, done] = evaluateTrajectory(t);
            if done
                if isempty(fruit_queue)
                    state = 'DONE';
                    status_msg = sprintf('All %d fruits sorted!', sorted_count);
                else
                    state = 'SELECT_FRUIT';
                    status_msg = 'Selecting next fruit';
                end
            else
                [joint_cmd, ~] = qarm_IK(pos, 0);
                gripper_cmd = 0;
                state_out = state;
                status_msg = 'Moving to home...';
                return;
            end

        case 'SELECT_FRUIT'
            % Pick nearest fruit from queue
            if isempty(fruit_queue)
                state = 'DONE';
            else
                current_target = fruit_queue{1};
                fruit_queue(1) = [];

                % Determine basket
                switch current_target.type
                    case 'strawberry'
                        target_basket = basket_positions.strawberry;
                    case 'banana'
                        target_basket = basket_positions.banana;
                    case 'tomato'
                        target_basket = basket_positions.tomato;
                    otherwise
                        target_basket = basket_positions.tomato;
                end
                state = 'APPROACH_FRUIT';
                status_msg = sprintf('Targeting %s', current_target.type);
            end

        case 'APPROACH_FRUIT'
            % Move above the fruit
            approach_pos = current_target.pos;
            approach_pos(3) = approach_z;
            wp = [ee_pos, approach_pos];
            dur = T_transit;
            setupTrajectory(wp, dur, t);
            state = 'MOVING_TO_APPROACH';
            status_msg = sprintf('Approaching %s', current_target.type);

        case 'MOVING_TO_APPROACH'
            [pos, done] = evaluateTrajectory(t);
            if done
                state = 'DESCEND_TO_PICK';
            else
                [joint_cmd, ~] = qarm_IK(pos, 0);
                gripper_cmd = 0;
                state_out = state;
                status_msg = 'Approaching fruit...';
                return;
            end

        case 'DESCEND_TO_PICK'
            % Descend to pick height
            pick_pos = current_target.pos;
            pick_pos(3) = pick_z;
            wp = [ee_pos, pick_pos];
            dur = T_pick;
            setupTrajectory(wp, dur, t);
            state = 'DESCENDING';
            gripper_state = 0;  % Ensure gripper open
            status_msg = 'Descending to pick...';

        case 'DESCENDING'
            [pos, done] = evaluateTrajectory(t);
            if done
                state = 'CLOSE_GRIPPER';
                traj_start_time = t;  % Reuse for dwell timing
            else
                [joint_cmd, ~] = qarm_IK(pos, 0);
                gripper_cmd = 0;
                state_out = state;
                status_msg = 'Descending...';
                return;
            end

        case 'CLOSE_GRIPPER'
            gripper_state = 1;
            if (t - traj_start_time) >= T_dwell
                state = 'ASCEND_FROM_PICK';
                status_msg = 'Gripper closed, ascending';
            else
                [joint_cmd, ~] = qarm_IK(ee_pos, 0);
                gripper_cmd = 1;
                state_out = state;
                status_msg = 'Closing gripper...';
                return;
            end

        case 'ASCEND_FROM_PICK'
            ascend_pos = ee_pos;
            ascend_pos(3) = safe_z;
            wp = [ee_pos, ascend_pos];
            dur = T_approach;
            setupTrajectory(wp, dur, t);
            state = 'ASCENDING_PICK';
            status_msg = 'Ascending with fruit';

        case 'ASCENDING_PICK'
            [pos, done] = evaluateTrajectory(t);
            if done
                state = 'MOVE_TO_BASKET';
            else
                [joint_cmd, ~] = qarm_IK(pos, 0);
                gripper_cmd = 1;
                state_out = state;
                status_msg = 'Ascending...';
                return;
            end

        case 'MOVE_TO_BASKET'
            basket_above = target_basket;
            basket_above(3) = safe_z;
            wp = [ee_pos, basket_above];
            dur = T_transit;
            setupTrajectory(wp, dur, t);
            state = 'MOVING_TO_BASKET';
            status_msg = sprintf('Moving to %s basket', current_target.type);

        case 'MOVING_TO_BASKET'
            [pos, done] = evaluateTrajectory(t);
            if done
                state = 'DESCEND_TO_PLACE';
            else
                [joint_cmd, ~] = qarm_IK(pos, 0);
                gripper_cmd = 1;
                state_out = state;
                status_msg = 'Moving to basket...';
                return;
            end

        case 'DESCEND_TO_PLACE'
            place_pos = target_basket;
            place_pos(3) = place_z;
            wp = [ee_pos, place_pos];
            dur = T_approach;
            setupTrajectory(wp, dur, t);
            state = 'DESCENDING_PLACE';
            status_msg = 'Descending to place';

        case 'DESCENDING_PLACE'
            [pos, done] = evaluateTrajectory(t);
            if done
                state = 'OPEN_GRIPPER';
                traj_start_time = t;
            else
                [joint_cmd, ~] = qarm_IK(pos, 0);
                gripper_cmd = 1;
                state_out = state;
                status_msg = 'Descending to basket...';
                return;
            end

        case 'OPEN_GRIPPER'
            gripper_state = 0;
            if (t - traj_start_time) >= T_dwell
                sorted_count = sorted_count + 1;
                state = 'ASCEND_FROM_PLACE';
                status_msg = sprintf('Placed %s (#%d)', current_target.type, sorted_count);
            else
                [joint_cmd, ~] = qarm_IK(ee_pos, 0);
                gripper_cmd = 0;
                state_out = state;
                status_msg = 'Opening gripper...';
                return;
            end

        case 'ASCEND_FROM_PLACE'
            ascend_pos = ee_pos;
            ascend_pos(3) = safe_z;
            wp = [ee_pos, ascend_pos];
            dur = T_approach;
            setupTrajectory(wp, dur, t);
            state = 'ASCENDING_PLACE';
            status_msg = 'Ascending from basket';

        case 'ASCENDING_PLACE'
            [pos, done] = evaluateTrajectory(t);
            if done
                state = 'GO_HOME';
            else
                [joint_cmd, ~] = qarm_IK(pos, 0);
                gripper_cmd = 0;
                state_out = state;
                status_msg = 'Ascending...';
                return;
            end

        case 'DONE'
            [joint_cmd, ~] = qarm_IK(home_pos, 0);
            gripper_cmd = 0;
            state_out = 'DONE';
            status_msg = sprintf('Complete! Sorted %d fruits.', sorted_count);
            return;

        otherwise
            state = 'INIT';
            status_msg = 'Unknown state, resetting';
    end

    % Default outputs for transition states
    [joint_cmd, ~] = qarm_IK(ee_pos, 0);
    gripper_cmd = gripper_state;

    %% ---- Nested helper functions ----
    function setupTrajectory(waypoints, duration, time_now)
        traj_waypoints = waypoints;
        if isscalar(duration)
            traj_durations = duration;
        else
            traj_durations = duration;
        end
        traj_total_time = sum(traj_durations);
        traj_start_time = time_now;
    end

    function [pos, done] = evaluateTrajectory(time_now)
        elapsed = time_now - traj_start_time;
        if elapsed >= traj_total_time
            pos = traj_waypoints(:, end);
            done = true;
        else
            if size(traj_waypoints, 2) == 2
                pos = cubicTrajectory(traj_waypoints(:,1), traj_waypoints(:,2), ...
                    traj_total_time, elapsed);
            else
                [pos, ~, ~] = multiSegmentTrajectory(traj_waypoints, traj_durations, elapsed);
            end
            done = false;
        end
    end
end
