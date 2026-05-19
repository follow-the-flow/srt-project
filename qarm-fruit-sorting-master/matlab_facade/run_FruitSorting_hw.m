function run_FruitSorting_hw()
%RUN_FRUITSORTING_HW Hardware launcher for FruitSorting_Hardware.slx.
%   Loads the top-level model, flips the qarm_mode Constant to 1
%   (real QArm via py.qarm_driver), runs sim() with real-time pacing
%   inside py_qarm_io, and guarantees the driver handle is released
%   even if the simulation errors out.

    facade_dir = fileparts(mfilename('fullpath'));
    addpath(facade_dir);
    setup_pyenv();
    clear py_controller py_qarm_io

    model = 'FruitSorting_Hardware';
    mdl_path = fullfile(facade_dir, [model '.slx']);
    if ~isfile(mdl_path)
        build_hardware_model();
    end

    load_system(mdl_path);

    prev_mode = get_param([model '/qarm_mode'], 'Value');
    set_param([model '/qarm_mode'], 'Value', '1');
    fprintf('qarm_mode: %s -> 1 (HARDWARE)\n', prev_mode);

    cleanupObj = onCleanup(@() hw_cleanup(model, prev_mode));

    fprintf('Starting hardware sim (expect ~30 s wall-clock)...\n');
    t0 = tic;
    out = sim(model);
    fprintf('sim() returned in %.2f s wall-clock\n', toc(t0));

    state = squeeze(out.get('state_log'));
    done  = squeeze(out.get('done_log'));
    phi   = squeeze(out.get('phi_cmd_log'));
    fb    = squeeze(out.get('joints_fb_log'));
    if size(phi,1) == 4, phi = phi.'; end
    if size(fb,1) == 4,  fb  = fb.';  end

    fprintf('\n--- RESULT ---\n');
    fprintf('steps          : %d\n', numel(state));
    fprintf('states visited : %s\n', num2str(unique(state(:))'));
    fprintf('final state    : %d  (11 = DONE)\n', state(end));
    fprintf('done flag      : %d\n', done(end));
    fprintf('max |fb - cmd| : %.3e rad\n', max(abs(fb(:) - phi(:))));

    if state(end) == 11 && done(end) == 1
        fprintf('\n=== FruitSorting_Hardware: HARDWARE RUN OK ===\n');
    else
        warning('FSM did not reach DONE on hardware run.');
    end
end

function hw_cleanup(model, prev_mode)
    try
        py_qarm_io(zeros(4,1), 0, 2);  % release driver
    catch ME
        fprintf('py_qarm_io release failed: %s\n', ME.message);
    end
    try
        set_param([model '/qarm_mode'], 'Value', prev_mode);
        save_system(model);
    catch
    end
    try
        close_system(model, 0);
    catch
    end
    fprintf('hw_cleanup: driver released, qarm_mode restored to %s\n', prev_mode);
end
