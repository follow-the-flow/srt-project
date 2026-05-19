function build_hardware_model_v2()
%BUILD_HARDWARE_MODEL_V2 Unified FruitSorting_Hardware model with a
%3-way mode switch (autonomous / remote / release) feeding a single
%py_qarm_io dispatcher. Rebuilds the .slx from scratch.

    modelName = 'FruitSorting_Hardware';
    if bdIsLoaded(modelName), close_system(modelName, 0); end
    new_system(modelName); open_system(modelName);

    % --- Mode selector ---
    add_block('simulink/Sources/Constant', [modelName '/mode'], ...
              'Value', '0', 'Position', [30, 20, 80, 40]);
    add_block('simulink/Signal Routing/Multiport Switch', ...
              [modelName '/mode_select'], ...
              'Inputs', '3', 'Position', [400, 40, 450, 180]);

    % --- Autonomous branch: Stateflow model reference ---
    add_block('simulink/Model-Wide Utilities/Model Info', ...
              [modelName '/autonomous'], 'Position', [200, 20, 320, 50]);
    % In practice: a Model Reference block to FruitSorting_Autonomous.slx.
    % For v2 scaffold we add a placeholder and wire it in when the
    % Stateflow chart build completes (Task A7.3).

    % --- Remote branch: HMI subsystem ---
    add_block('simulink/Ports & Subsystems/Subsystem', ...
              [modelName '/remote_hmi'], 'Position', [200, 80, 320, 110]);
    % Populate inside with jog buttons (Push Buttons from Dashboard lib),
    % mode sub-toggle, gripper open/close buttons, E-stop, workspace
    % clamp via MATLAB Function block calling py.remote_jog.* helpers.
    % Full wiring detailed in remote_hmi.md (section below).

    % --- Release branch: zeros ---
    add_block('simulink/Sources/Constant', [modelName '/release_cmd'], ...
              'Value', 'zeros(4,1)', 'Position', [200, 140, 320, 170]);

    % --- Shared output: py_qarm_io ---
    add_block('simulink/User-Defined Functions/MATLAB Function', ...
              [modelName '/py_qarm_io'], 'Position', [500, 60, 620, 120]);
    % py_qarm_io expects (phi_cmd, gripper_cmd, mode) -- wire mode from
    % the same Constant as mode_select so SIMULATE/HARDWARE/RELEASE
    % toggling is exposed to the evaluator.

    % E-stop button (Dashboard Push Button, signal-routes to Stop block
    % via Stateflow state forcing): added in a follow-up commit.

    % Logging: add 'To Workspace' blocks on commanded phi, gripper,
    % state_id, joints_read so generate_report_plots.py can find the
    % .mat at end of run.
    for i = 1:4
        nm = sprintf('log_joint_%d', i);
        add_block('simulink/Sinks/To Workspace', [modelName '/' nm], ...
                  'VariableName', nm, 'SaveFormat', 'StructureWithTime', ...
                  'Position', [700, 20 + (i-1)*40, 780, 40 + (i-1)*40]);
    end

    save_system(modelName);
    close_system(modelName);
    fprintf('Built %s.slx (v2, unified 3-way model).\n', modelName);
end
