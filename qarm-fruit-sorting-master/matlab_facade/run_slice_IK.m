function run_slice_IK()
%RUN_SLICE_IK Build and simulate the IK vertical slice, print captured joints.

    facade_dir = fileparts(mfilename('fullpath'));
    addpath(facade_dir);

    setup_pyenv();

    if ~isfile(fullfile(facade_dir, 'FruitSorting_slice.slx'))
        build_slice_IK();
    end

    model = 'FruitSorting_slice';
    load_system(fullfile(facade_dir, [model '.slx']));
    out = sim(model);
    close_system(model, 0);

    phi = out.get('phi_log');
    phi = squeeze(phi)';                 % N x 4
    fprintf('phi_log samples: %d\n', size(phi,1));
    last = phi(end,:);
    fprintf('last phi (rad): [%s]\n', num2str(last, '% .4f'));

    expected = [-0.5880 0.5059 0.6880 0.0000];
    err = max(abs(last - expected));
    fprintf('max deviation from expected: %.6f rad\n', err);
    assert(err < 1e-3, 'IK slice produced unexpected joint vector');
    disp('SLICE 1 PASS');
end
