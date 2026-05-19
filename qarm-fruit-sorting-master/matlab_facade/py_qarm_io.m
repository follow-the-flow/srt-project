function [joints_read, gripper_read] = py_qarm_io(phi_cmd, gripper_cmd, mode)
%PY_QARM_IO Persistent facade over py.qarm_driver.QArmDriver.
%   mode : 0 = SIMULATE (echo commanded joints back, no hardware)
%          1 = HARDWARE (open card, issue command, read back, real-time pace)
%          2 = RELEASE  (disconnect and clear the persistent handle)
%
%   In mode 1 the call blocks until the wall-clock matches solver dt
%   (DT_TARGET below), so Simulink's non-real-time sim() advances at the
%   pace the FSM trajectories were designed for.

    persistent drv tick_timer
    DT_TARGET = 0.05;   % must match build_hardware_model.m dt

    joints_read = zeros(4,1);
    gripper_read = 0;

    if mode == 2
        if ~isempty(drv)
            try
                drv.disconnect();
            catch
            end
            drv = [];
        end
        tick_timer = [];
        return;
    end

    if mode == 0
        joints_read = phi_cmd(:);
        gripper_read = gripper_cmd;
        return;
    end

    % ---- Hardware path ----
    if isempty(drv)
        drv = py.qarm_driver.QArmDriver();
        drv.connect();
        tick_timer = tic;
    end

    drv.set_joints_and_gripper(py.numpy.array(phi_cmd(:)'), gripper_cmd);
    result = drv.read_all();
    joints_read = double(result{1})';
    joints_read = joints_read(:);
    gripper_read = double(result{2});

    % Real-time pacing: sleep until DT_TARGET has elapsed since last tick.
    elapsed = toc(tick_timer);
    if elapsed < DT_TARGET
        pause(DT_TARGET - elapsed);
    end
    tick_timer = tic;
end
