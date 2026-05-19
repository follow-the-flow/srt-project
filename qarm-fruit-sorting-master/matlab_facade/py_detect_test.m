function [det, count] = py_detect_test(trigger) %#ok<INUSD>
%PY_DETECT_TEST Call py.fruit_detector on a synthetic BGR image.
%   Used by the vision vertical slice — Simulink has no easy way to feed
%   a uint8 HxWx3 image into a MATLAB Function block, so the image is
%   generated here. Returns a fixed-size detection matrix.
%
%   det   : 8 x 5 double, rows = [type_id, cy, cx, area, conf]
%           type_id: 1 strawberry, 2 tomato, 3 banana, 0 empty row
%   count : scalar int32

    N_MAX = int32(8);
    det = zeros(N_MAX, 5);
    count = int32(0);

    bgr = zeros(240, 320, 3, 'uint8');
    % Yellow square (BGR [0 255 255]) -> banana
    bgr(40:100, 40:100, 2) = 255;
    bgr(40:100, 40:100, 3) = 255;
    % Red square 1 (BGR [0 0 255]) -> compact/round -> tomato
    bgr(150:210, 60:120, 3) = 255;
    % Red very-elongated rectangle -> low circularity -> strawberry
    bgr(130:220, 210:225, 3) = 255;

    pyImg = py.numpy.array(bgr);
    result = py.fruit_detector.detect_fruits(pyImg);

    n = int32(py.len(result));
    if n > N_MAX, n = N_MAX; end

    type_map = struct('strawberry', 1, 'tomato', 2, 'banana', 3);
    for i = 1:double(n)
        d = result{i};
        tname = char(d.fruit_type);
        if isfield(type_map, tname)
            tid = type_map.(tname);
        else
            tid = 0;
        end
        centroid = cell(d.centroid);
        cy = double(centroid{1});
        cx = double(centroid{2});
        area = double(d.area);
        conf = double(d.confidence);
        det(i, :) = [tid, cy, cx, area, conf];
    end
    count = n;
end
