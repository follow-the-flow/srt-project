function build_autonomous_stateflow()
%BUILD_AUTONOMOUS_STATEFLOW Assemble the Stateflow chart that drives
%the autonomous FSM. 12 states, transitions timed by dt accumulator.
%Entry actions call py.sorting_controller.* helpers for the heavy work
%(IK safety, gripper ramp+readback, smart pick_z, pre-flight).

    modelName = 'FruitSorting_Autonomous';
    if bdIsLoaded(modelName), close_system(modelName, 0); end
    new_system(modelName); open_system(modelName);

    % Inputs and outputs
    add_block('simulink/Sources/In1', [modelName '/joints_cur'], ...
              'Position', [20, 20, 50, 40]);
    add_block('simulink/Sources/In1', [modelName '/dt'], ...
              'Position', [20, 80, 50, 100]);
    add_block('simulink/Sinks/Out1', [modelName '/phi'], ...
              'Position', [520, 20, 550, 40]);
    add_block('simulink/Sinks/Out1', [modelName '/gripper'], ...
              'Position', [520, 60, 550, 80]);
    add_block('simulink/Sinks/Out1', [modelName '/state_id'], ...
              'Position', [520, 100, 550, 120]);
    add_block('simulink/Sinks/Out1', [modelName '/done'], ...
              'Position', [520, 140, 550, 160]);

    % Insert Stateflow chart
    chart = Stateflow.Chart(get_param(modelName, 'Object'));
    chart.Name = 'FruitSortFSM';
    % Chart I/O
    for nm = {'joints_cur', 'dt'}
        d = Stateflow.Data(chart); d.Name = nm{1}; d.Scope = 'Input';
    end
    for nm = {'phi', 'gripper', 'state_id', 'done'}
        d = Stateflow.Data(chart); d.Name = nm{1}; d.Scope = 'Output';
    end
    % Book-keeping variables
    for nm = {'t_transit', 'sorted_count'}
        d = Stateflow.Data(chart); d.Name = nm{1}; d.Scope = 'Local';
    end

    states = struct( ...
        'INIT',         [30, 30,  120, 80], ...
        'GO_HOME',      [180, 30, 300, 80], ...
        'SELECT',       [360, 30, 480, 80], ...
        'APPROACH',     [540, 30, 660, 80], ...
        'DESCEND',      [540, 120, 660, 170], ...
        'CLOSE_GRIP',   [540, 210, 660, 260], ...
        'ASCEND_PICK',  [360, 210, 480, 260], ...
        'MOVE_BASKET',  [180, 210, 300, 260], ...
        'DESCEND_PLACE',[30, 210, 150, 260], ...
        'OPEN_GRIP',    [30, 300, 150, 350], ...
        'ASCEND_PLACE', [180, 300, 300, 350], ...
        'DONE',         [360, 300, 480, 350]);
    names = fieldnames(states);
    state_objs = struct();
    for k = 1:numel(names)
        nm = names{k};
        s = Stateflow.State(chart);
        s.Name = nm;
        s.Position = states.(nm);
        state_objs.(nm) = s;
    end

    % Entry actions — each calls the matching Python helper so logic stays
    % one-source-of-truth in python/sorting_controller.py.
    % Note: exact MATLAB<->Python call syntax (double() wrapping, py.list
    % conversions) may need tweaking during first MATLAB run.
    state_objs.SELECT.EntryAction = ...
        'phi = double(py.sorting_controller.stateflow_select(joints_cur))';
    state_objs.CLOSE_GRIP.EntryAction = ...
        'gripper = double(py.sorting_controller.stateflow_close_grip())';
    state_objs.OPEN_GRIP.EntryAction = ...
        'gripper = double(py.sorting_controller.stateflow_open_grip())';

    % Default transition
    def_tr = Stateflow.Transition(chart);
    def_tr.Destination = state_objs.INIT;
    def_tr.SourceEndpoint = [30, 5];
    def_tr.DestinationEndpoint = [30, 30];

    % Subsequent transitions (INIT -> GO_HOME -> SELECT -> APPROACH -> ...)
    pairs = {
        'INIT','GO_HOME','t_transit > 2.0';
        'GO_HOME','SELECT','~isempty(py.sorting_controller.fruit_queue())';
        'SELECT','APPROACH','true';
        'APPROACH','DESCEND','t_transit > 2.0';
        'DESCEND','CLOSE_GRIP','t_transit > 0.8';
        'CLOSE_GRIP','ASCEND_PICK','t_transit > 0.8';
        'ASCEND_PICK','MOVE_BASKET','t_transit > 1.0';
        'MOVE_BASKET','DESCEND_PLACE','t_transit > 2.0';
        'DESCEND_PLACE','OPEN_GRIP','t_transit > 1.0';
        'OPEN_GRIP','ASCEND_PLACE','t_transit > 0.8';
        'ASCEND_PLACE','GO_HOME','t_transit > 1.0';
        'GO_HOME','DONE','isempty(py.sorting_controller.fruit_queue())'};
    for r = 1:size(pairs, 1)
        tr = Stateflow.Transition(chart);
        tr.Source = state_objs.(pairs{r, 1});
        tr.Destination = state_objs.(pairs{r, 2});
        tr.LabelString = pairs{r, 3};
    end

    save_system(modelName);
    close_system(modelName);
    fprintf('Built %s.slx\n', modelName);
end
