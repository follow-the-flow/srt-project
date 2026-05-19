function run_FruitSorting()
%RUN_FRUITSORTING Launcher for the full FruitSorting_Hardware.slx model.
%   Simulate path by default (qarm_mode=0). To run against the real
%   QArm, open the model after build and set the 'qarm_mode' Constant
%   block value to 1. Requires QUARC-compatible hardware access through
%   py.qarm_driver.QArmDriver on the lab machine.

    facade_dir = fileparts(mfilename('fullpath'));
    addpath(facade_dir);
    setup_pyenv();
    clear py_controller py_qarm_io

    if ~isfile(fullfile(facade_dir, 'FruitSorting_Hardware.slx'))
        build_hardware_model();
    end

    model = 'FruitSorting_Hardware';
    load_system(fullfile(facade_dir, [model '.slx']));
    out = sim(model);
    close_system(model, 0);

    state = squeeze(out.get('state_log'));
    done  = squeeze(out.get('done_log'));
    phi   = squeeze(out.get('phi_cmd_log'));
    fb    = squeeze(out.get('joints_fb_log'));

    if size(phi,1) == 4, phi = phi.'; end
    if size(fb,1) == 4,  fb  = fb.';  end

    fprintf('steps          : %d\n', numel(state));
    fprintf('states visited : %s\n', num2str(unique(state(:))'));
    fprintf('final state    : %d  (11 = DONE)\n', state(end));
    fprintf('done flag      : %d\n', done(end));
    fprintf('max |fb - cmd| : %.3e rad\n', max(abs(fb(:) - phi(:))));

    assert(state(end) == 11, 'FSM did not reach DONE');
    assert(done(end) == 1,   'done flag not set');
    assert(all(ismember(1:11, unique(state(:))')), ...
        'Not all FSM states exercised');
    fprintf('\n=== FruitSorting_Hardware: SIMULATION PASS ===\n');
end
