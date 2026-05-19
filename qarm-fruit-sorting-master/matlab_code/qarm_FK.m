function [p4, R04] = qarm_FK(phi)
%QARM_FK Forward kinematics for QArm (reusable standalone version)
%   INPUT:  phi - 4x1 joint angles [phi1; phi2; phi3; phi4] in radians
%   OUTPUT: p4  - 3x1 end-effector position [px; py; pz] in metres
%           R04 - 3x3 rotation matrix (end-effector orientation)

    l1 = 0.14;
    l2 = sqrt(0.35^2 + 0.05^2);   % 0.3536 m
    l3 = 0.25 + 0.15;             % 0.40 m
    beta = atan(0.05 / 0.35);

    theta = zeros(4, 1);
    theta(1) = phi(1);
    theta(2) = phi(2) - pi/2 + beta;
    theta(3) = phi(3) - beta;
    theta(4) = phi(4);

    T01 = dh_matrix(0,   -pi/2,  l1,  theta(1));
    T12 = dh_matrix(l2,   0,      0,   theta(2));
    T23 = dh_matrix(0,   -pi/2,   0,   theta(3));
    T34 = dh_matrix(0,    0,      l3,  theta(4));

    T04 = T01 * T12 * T23 * T34;
    p4  = T04(1:3, 4);
    R04 = T04(1:3, 1:3);
end

function T = dh_matrix(a, alpha, d, theta)
    ct = cos(theta); st = sin(theta);
    ca = cos(alpha); sa = sin(alpha);
    T = [ct  -st*ca   st*sa   a*ct;
         st   ct*ca  -ct*sa   a*st;
         0    sa      ca      d;
         0    0       0       1];
end
