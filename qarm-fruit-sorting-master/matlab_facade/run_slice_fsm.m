function run_slice_fsm()
%RUN_SLICE_FSM Build and simulate the full closed-loop FSM slice.

    facade_dir = fileparts(mfilename('fullpath'));
    addpath(facade_dir);
    setup_pyenv();
    clear py_controller

    if ~isfile(fullfile(facade_dir, 'FruitSorting_slice_fsm.slx'))
        build_slice_fsm();
    end

    model = 'FruitSorting_slice_fsm';
    load_system(fullfile(facade_dir, [model '.slx']));
    out = sim(model);
    close_system(model, 0);

    phi   = squeeze(out.get('phi_cmd_log'));
    grip  = squeeze(out.get('gripper_log'));
    state = squeeze(out.get('state_log'));
    done  = squeeze(out.get('done_log'));

    if size(phi,1) == 4, phi = phi.'; end   % force N x 4

    fprintf('steps: %d\n', numel(state));
    fprintf('unique states visited: %s\n', num2str(unique(state(:))'));
    fprintf('final state: %d, done flag: %d\n', state(end), done(end));
    fprintf('gripper toggles: %d\n', sum(abs(diff(grip(:))) > 0.5));

    assert(any(state == 11), 'FSM never reached DONE state');
    assert(all(ismember(1:11, unique(state(:))')), ...
        'FSM did not visit every state');
    assert(done(end) == 1, 'done flag not set at end of sim');
    disp('SLICE 3 PASS');
end
