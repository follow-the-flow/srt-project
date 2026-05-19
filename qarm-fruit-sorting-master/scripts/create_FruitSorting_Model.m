%% CREATE_FRUITSORTING_MODEL
% Programmatically creates the FruitSorting Simulink model for QArm
% This script builds the complete model for HARDWARE deployment.
%
% RUN THIS SCRIPT ONCE to generate FruitSorting_Hardware.slx
% Then open the model in Simulink for further customization.
%
% Required Toolboxes:
%   - Simulink, Stateflow, Robotics System Toolbox
%   - Computer Vision Toolbox, Image Processing Toolbox

%% Setup
modelName = 'FruitSorting_Hardware';
close_system(modelName, 0);  % Close if open (don't save)
if exist([modelName '.slx'], 'file')
    delete([modelName '.slx']);
end

% Create new model
new_system(modelName);
open_system(modelName);

%% --- Configuration Parameters ---
cs = getActiveConfigSet(gcs);
set_param(cs, 'SolverType', 'Fixed-step');
set_param(cs, 'Solver', 'ode4');
set_param(cs, 'FixedStep', '0.002');   % 500 Hz
set_param(cs, 'StopTime', '300');       % 5 minutes max

%% --- Add QArm Hardware Interface Block ---
% NOTE: The Quanser QUARC blocks must be added manually if QUARC is installed.
% This creates placeholder subsystems that match the QUARC interface.

% ====== QArm Plant Subsystem ======
add_block('simulink/Ports & Subsystems/Subsystem', ...
    [modelName '/QArm Plant'], ...
    'Position', [400, 150, 550, 350]);

% Inside QArm Plant: placeholders for QUARC HIL blocks
% (Will be replaced with actual QUARC blocks on hardware PC)

% ====== Mode Switch (Auto/Remote) ======
add_block('simulink/Signal Routing/Manual Switch', ...
    [modelName '/Mode Switch'], ...
    'Position', [250, 200, 280, 240]);

% ====== Autonomous Controller Subsystem ======
add_block('simulink/Ports & Subsystems/Subsystem', ...
    [modelName '/Autonomous Controller'], ...
    'Position', [100, 100, 250, 200]);

% ====== Remote Controller Subsystem ======
add_block('simulink/Ports & Subsystems/Subsystem', ...
    [modelName '/Remote Controller'], ...
    'Position', [100, 280, 250, 380]);

% ====== Vision System Subsystem ======
add_block('simulink/Ports & Subsystems/Subsystem', ...
    [modelName '/Vision System'], ...
    'Position', [100, 420, 250, 520]);

% ====== Forward Kinematics Block ======
add_block('simulink/User-Defined Functions/MATLAB Function', ...
    [modelName '/FK'], ...
    'Position', [600, 200, 700, 250]);

% ====== Scopes ======
add_block('simulink/Sinks/Scope', ...
    [modelName '/Joint Tracking'], ...
    'Position', [750, 150, 800, 190]);
add_block('simulink/Sinks/Scope', ...
    [modelName '/EE Position'], ...
    'Position', [750, 220, 800, 260]);
add_block('simulink/Sinks/Scope', ...
    [modelName '/State Monitor'], ...
    'Position', [750, 290, 800, 330]);

% ====== Display blocks ======
add_block('simulink/Sinks/Display', ...
    [modelName '/Current State'], ...
    'Position', [750, 360, 850, 390]);
add_block('simulink/Sinks/Display', ...
    [modelName '/Fruits Sorted'], ...
    'Position', [750, 410, 850, 440]);

% ====== Clock ======
add_block('simulink/Sources/Clock', ...
    [modelName '/Clock'], ...
    'Position', [30, 160, 60, 180]);

%% --- Save Model ---
save_system(modelName);
fprintf('Model "%s.slx" created successfully.\n', modelName);
fprintf('\nNEXT STEPS:\n');
fprintf('1. Open the model in Simulink\n');
fprintf('2. Replace "QArm Plant" subsystem with QUARC QArm blocks\n');
fprintf('3. Add MATLAB Function blocks with the provided .m files\n');
fprintf('4. Connect the signal paths as described in the README\n');
fprintf('5. Build and deploy with QUARC\n');
