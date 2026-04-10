function [p4, R04] = qarmForwardKinematics(phi)

%% QARM_FORWARD_KINEMATICS
% Forward kinematics for QArm manipulator using Standard DH parameters
% Based on Quanser Lab 2 - Concept Review

% INPUTS:
% phi     : Joint angles vector 4 x 1 [rad] (physical joint space)

% OUTPUTS:
% p4      : End-effector position in base frame {0} [3 x 1]
% R04     : Rotation matrix from {4} to {0} [3 x 3]

%% Manipulator physical parameters (meters)
L1 = 0.1400;
L2 = 0.3500;
L3 = 0.0500;
L4 = 0.2500;
L5 = 0.1500;

l1 = L1;
l2 = sqrt(L2^2+L3^2);
l3 = L4 + L5;
beta = atan2(L3, L2);                 % β = tan⁻¹(L3/L2)

%% Map from phi space to theta space (Table 2)
theta = zeros(4, 1);
theta(1) = phi(1);                                    % θ1 = φ1
theta(2) = phi(2) + pi/2 - beta;                      % θ2 = φ2 + π/2 - β
theta(3) = phi(3) + beta;                             % θ3 = φ3 + β
theta(4) = phi(4);                                    % θ4 = φ4

%% DH transformation matrices
T01 = dhTransform(0,  pi/2, l1, theta(1)); % 基座到连杆 1
T12 = dhTransform(l2, 0,    0,  theta(2)); % 连杆 1 到连杆 2
T23 = dhTransform(0,  pi/2, 0,  theta(3)); % 连杆 2 到连杆 3
T34 = dhTransform(0,  0,    l3, theta(4)); % 连杆 3 到末端执行器

%% Compound transformations
T02 = T01 * T12;
T03 = T02 * T23;
T04 = T03 * T34;

%% Extract position and rotation
p4 = T04(1:3, 4);
R04 = T04(1:3, 1:3);

end

%% ------------------------------------------------------------------------
function T = dhTransform(a, alpha, d, theta)
% Standard DH transformation matrix from frame {i} to frame {i-1}
% T = Rot_z(theta) * Trans_z(d) * Trans_x(a) * Rot_x(alpha)

% Rotation about z by theta
Rz = [cos(theta), -sin(theta), 0, 0;
      sin(theta),  cos(theta), 0, 0;
      0,           0,          1, 0;
      0,           0,          0, 1];

% Translation along z by d
Tz = [1, 0, 0, 0;
      0, 1, 0, 0;
      0, 0, 1, d;
      0, 0, 0, 1];

% Translation along x by a
Tx = [1, 0, 0, a;
      0, 1, 0, 0;
      0, 0, 1, 0;
      0, 0, 0, 1];

% Rotation about x by alpha
Rx = [1, 0,           0,          0;
      0, cos(alpha), -sin(alpha), 0;
      0, sin(alpha),  cos(alpha), 0;
      0, 0,           0,          1];

% Combine
T = Rz * Tz * Tx * Rx;

end