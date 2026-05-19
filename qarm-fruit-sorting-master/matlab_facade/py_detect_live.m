function [det, count] = py_detect_live(trigger, calib_path) %#ok<INUSD>
%PY_DETECT_LIVE Open the D415, capture one frame, run fruit_detector,
%project each detection into the robot base frame, and return a fixed-size
%matrix suitable for a Simulink MATLAB Function block.
%
%   Calibration: reads the T_cam_to_base matrix + camera intrinsics from
%   calib_path (default '../calibration.json'). Because the D415 is arm-
%   mounted, the arm must be at the pose where calib_path was derived
%   before this block is triggered — use pickhome1 by default, or the
%   pose label recorded under calib_path.pose_label.
%
%   Inputs
%   ------
%   trigger    : scalar, rising edge fires a capture. Value is unused —
%                a pulse gate in the model handles one-shot semantics.
%   calib_path : char row vector, path to calibration JSON. Empty or
%                missing -> '../calibration.json'.
%
%   Outputs
%   -------
%   det   : 16 x 5 double. Rows = [type_id, wx_m, wy_m, wz_m, conf].
%           type_id: 1 strawberry, 2 tomato, 3 banana, 0 empty row.
%   count : int32, number of populated rows (0 <= count <= 16).

    persistent cam intr T n_max
    N_MAX = 16;
    n_max = N_MAX;
    det = zeros(N_MAX, 5);
    count = int32(0);

    if nargin < 2 || isempty(calib_path)
        calib_path = fullfile(fileparts(mfilename('fullpath')), '..', ...
                              'calibration.json');
    end

    % Lazy-open the camera once
    if isempty(cam)
        cam = py.camera.QArmCamera();
        cam.open();
        intr_py = cam.intrinsics;
        intr = struct( ...
            'fx', double(intr_py{'fx'}), ...
            'fy', double(intr_py{'fy'}), ...
            'cx', double(intr_py{'cx'}), ...
            'cy', double(intr_py{'cy'}));
        % Warmup until a non-black color frame with depth coverage arrives
        deadline = tic;
        while toc(deadline) < 10
            try
                frame = cam.read();
                color = uint8(frame{1});
                depthU = uint16(frame{2});
                if mean(color(:)) > 5 && mean(depthU(:) > 0) > 0.10
                    break;
                end
            catch
            end
            pause(0.1);
        end
    end

    % Lazy-load calibration
    if isempty(T)
        if ~exist(calib_path, 'file')
            warning('py_detect_live:calibMissing', ...
                    'calibration file %s not found — using identity', ...
                    calib_path);
            T = eye(4);
        else
            txt = fileread(calib_path);
            cal = jsondecode(txt);
            T = cal.T_cam_to_base;
        end
    end

    % Capture + detect + project in one Python call
    frame = cam.read();
    color_py = frame{1};
    depth_py = frame{2};

    intr_py_dict = py.dict(pyargs( ...
        'fx', intr.fx, 'fy', intr.fy, 'cx', intr.cx, 'cy', intr.cy));
    T_py = py.numpy.array(T);

    result = py.fruit_detector.detect_and_project( ...
        color_py, depth_py, intr_py_dict, T_py, int32(N_MAX));

    out_py = result{1};
    n      = int32(result{2});
    if n > N_MAX, n = N_MAX; end

    mat = double(out_py);   % 16 x 5
    det = mat;
    count = n;
end
