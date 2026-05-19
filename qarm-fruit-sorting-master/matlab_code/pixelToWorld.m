function p_world = pixelToWorld(centroid_px, depth_image, cam_intrinsics, T_cam_to_base)
%PIXELTOWORLD Converts pixel coordinates + depth to QArm base frame coordinates
%   Uses the Intel RealSense D415 camera parameters.
%
%   INPUT:  centroid_px    - 1x2 pixel [row, col]
%           depth_image    - HxW depth map (metres or mm, auto-detected)
%           cam_intrinsics - struct with fields: fx, fy, cx, cy
%                           (RealSense D415 defaults: fx=fy=615, cx=320, cy=240)
%           T_cam_to_base  - 4x4 transformation from camera frame to QArm base
%   OUTPUT: p_world        - 3x1 position in QArm base frame [x; y; z]

    row = round(centroid_px(1));
    col = round(centroid_px(2));

    % Get depth at centroid (average 5x5 patch for noise robustness)
    [H, W] = size(depth_image);
    r_min = max(1, row-2); r_max = min(H, row+2);
    c_min = max(1, col-2); c_max = min(W, col+2);
    patch = depth_image(r_min:r_max, c_min:c_max);
    Z = median(patch(:), 'omitnan');

    % Auto-detect units: if > 10, assume mm, convert to metres
    if Z > 10
        Z = Z / 1000;
    end

    % Default RealSense D415 intrinsics (640x480 mode)
    if nargin < 3 || isempty(cam_intrinsics)
        cam_intrinsics.fx = 615;
        cam_intrinsics.fy = 615;
        cam_intrinsics.cx = 320;
        cam_intrinsics.cy = 240;
    end

    % Back-project pixel to camera frame
    X_cam = (col - cam_intrinsics.cx) * Z / cam_intrinsics.fx;
    Y_cam = (row - cam_intrinsics.cy) * Z / cam_intrinsics.fy;
    Z_cam = Z;

    p_cam = [X_cam; Y_cam; Z_cam; 1];

    % Default camera-to-base transform (must be calibrated for your setup!)
    if nargin < 4 || isempty(T_cam_to_base)
        % Typical QArm camera mount: camera at end-effector looking down
        % This is a placeholder - MUST be calibrated with the actual robot
        T_cam_to_base = eye(4);
        warning('pixelToWorld: Using identity transform. Calibrate T_cam_to_base for your setup.');
    end

    p_base = T_cam_to_base * p_cam;
    p_world = p_base(1:3);
end
