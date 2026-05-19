%% CALIBRATE_CAMERA - Camera-to-Robot Calibration for QArm + RealSense D415
% This script helps determine the transformation matrix T_cam_to_base
% that converts pixel+depth coordinates to QArm base frame coordinates.
%
% PROCEDURE:
% 1. Place a known calibration object (e.g., colored ball) at several
%    known positions in the QArm workspace
% 2. Record the pixel coordinates and depth from the camera
% 3. Record the actual world positions (using FK or known coordinates)
% 4. This script computes the optimal transformation
%
% The QArm's RealSense D415 is mounted at a FIXED position relative to
% the base. The transform is static once calibrated.

clear; clc;
fprintf('=== QArm Camera-to-Base Calibration ===\n\n');

%% --- Intel RealSense D415 Default Intrinsics (640x480) ---
% These should be verified with the actual camera using:
%   cam = realsense.pipeline(); profile = cam.start();
%   intrinsics = profile.get_stream(rs.stream.depth).as_video_stream_profile().get_intrinsics();
cam_intrinsics.fx = 615.0;
cam_intrinsics.fy = 615.0;
cam_intrinsics.cx = 320.0;
cam_intrinsics.cy = 240.0;

fprintf('Camera intrinsics (default D415 @ 640x480):\n');
fprintf('  fx=%.1f, fy=%.1f, cx=%.1f, cy=%.1f\n\n', ...
    cam_intrinsics.fx, cam_intrinsics.fy, cam_intrinsics.cx, cam_intrinsics.cy);

%% --- Calibration Points ---
% INSTRUCTIONS: Replace these with your actual measurements
% Format: [pixel_col, pixel_row, depth_m, world_x, world_y, world_z]
%
% STEPS TO COLLECT DATA:
%   1. Run ImageAcquisitionAndColorSpaces.slx from Lab 9
%   2. Place a brightly colored object at a KNOWN position
%   3. Read the pixel coordinates from the image display
%   4. Read the depth from the depth image
%   5. The known position is measured with a ruler or from IK validation
%   6. Repeat for at least 6 points spread across the workspace

% Example calibration data (REPLACE WITH YOUR MEASUREMENTS):
calib_data = [
%   px_col  px_row  depth_m  world_x   world_y   world_z
    320     240     0.350    0.250     0.000     0.050
    400     200     0.300    0.150     0.100     0.080
    250     280     0.400    0.300    -0.100     0.030
    350     180     0.280    0.200     0.050     0.100
    280     300     0.420    0.350    -0.050     0.020
    370     220     0.320    0.180     0.080     0.060
];

n_points = size(calib_data, 1);
fprintf('Using %d calibration points.\n', n_points);

%% --- Back-project pixels to camera frame ---
P_cam = zeros(3, n_points);
P_world = zeros(3, n_points);

for i = 1:n_points
    col = calib_data(i, 1);
    row = calib_data(i, 2);
    Z   = calib_data(i, 3);

    P_cam(1, i) = (col - cam_intrinsics.cx) * Z / cam_intrinsics.fx;
    P_cam(2, i) = (row - cam_intrinsics.cy) * Z / cam_intrinsics.fy;
    P_cam(3, i) = Z;

    P_world(:, i) = calib_data(i, 4:6)';
end

%% --- Solve for transformation T_cam_to_base ---
% We want: P_world = R * P_cam + t
% Using least squares with SVD

% Compute centroids
c_cam   = mean(P_cam, 2);
c_world = mean(P_world, 2);

% Center the points
P_cam_c   = P_cam - c_cam;
P_world_c = P_world - c_world;

% SVD to find rotation
H = P_cam_c * P_world_c';
[U, ~, V] = svd(H);
R = V * U';

% Ensure proper rotation (det = +1)
if det(R) < 0
    V(:, 3) = -V(:, 3);
    R = V * U';
end

% Translation
t_vec = c_world - R * c_cam;

% Build 4x4 homogeneous transform
T_cam_to_base = eye(4);
T_cam_to_base(1:3, 1:3) = R;
T_cam_to_base(1:3, 4) = t_vec;

%% --- Display Results ---
fprintf('\n=== Calibration Result ===\n');
fprintf('T_cam_to_base = \n');
disp(T_cam_to_base);

% Compute reprojection error
errors = zeros(1, n_points);
for i = 1:n_points
    p_est = T_cam_to_base * [P_cam(:,i); 1];
    errors(i) = norm(p_est(1:3) - P_world(:,i));
end

fprintf('Reprojection error (mm):\n');
fprintf('  Mean: %.2f mm\n', mean(errors)*1000);
fprintf('  Max:  %.2f mm\n', max(errors)*1000);
fprintf('  RMS:  %.2f mm\n', sqrt(mean(errors.^2))*1000);

%% --- Save Calibration ---
save('camera_calibration.mat', 'T_cam_to_base', 'cam_intrinsics', 'calib_data', 'errors');
fprintf('\nCalibration saved to camera_calibration.mat\n');
fprintf('\nUSAGE in your code:\n');
fprintf('  load(''camera_calibration.mat'');\n');
fprintf('  p_world = pixelToWorld(centroid_px, depth_image, cam_intrinsics, T_cam_to_base);\n');
