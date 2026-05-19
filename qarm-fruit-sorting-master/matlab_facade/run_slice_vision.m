function run_slice_vision()
%RUN_SLICE_VISION Build and simulate the vision vertical slice.

    facade_dir = fileparts(mfilename('fullpath'));
    addpath(facade_dir);
    setup_pyenv();

    if ~isfile(fullfile(facade_dir, 'FruitSorting_slice_vision.slx'))
        build_slice_vision();
    end

    model = 'FruitSorting_slice_vision';
    load_system(fullfile(facade_dir, [model '.slx']));
    out = sim(model);
    close_system(model, 0);

    det = squeeze(out.get('det_log'));
    cnt = squeeze(out.get('count_log'));
    if ndims(det) == 3
        det_last = det(:,:,end);
    else
        det_last = det;
    end
    if numel(cnt) > 1, cnt = cnt(end); end

    fprintf('count = %d\n', cnt);
    disp(det_last(1:double(cnt), :));

    types = det_last(1:double(cnt), 1);
    assert(cnt >= 3, 'expected at least 3 detections');
    assert(any(types == 1), 'expected a strawberry (type 1)');
    assert(any(types == 2), 'expected a tomato (type 2)');
    assert(any(types == 3), 'expected a banana (type 3)');
    disp('SLICE 2 PASS');
end
