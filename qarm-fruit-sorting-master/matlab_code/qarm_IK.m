function [phi_optimal, all_solutions] = qarm_IK(p, gamma)
%QARM_IK Inverse kinematics for QArm
%   Uses analytical solution + Newton-Raphson refinement for sub-mm accuracy.
%
%   INPUT:  p     - 3x1 desired position [px; py; pz] in metres
%           gamma - wrist angle (rad), typically 0
%   OUTPUT: phi_optimal    - 4x1 optimal joint angles (rad)
%           all_solutions  - 4x4 matrix of all analytical solutions

    % Step 1: Get analytical solution as initial guess
    [phi_guess, all_solutions] = analyticalIK(p, gamma);

    % Step 2: Refine with Newton-Raphson
    phi_optimal = refineIK(phi_guess, p, gamma);
end

function phi = refineIK(phi0, p_target, gamma)
%REFINEIK Newton-Raphson iterative refinement of IK solution
    phi = phi0;
    max_iter = 50;
    tol = 1e-6;  % 1 micron tolerance

    for iter = 1:max_iter
        [p_curr, ~] = qarm_FK(phi);
        err = p_target - p_curr;

        if norm(err) < tol
            break;
        end

        % Compute Jacobian numerically
        J = numericalJacobian(phi);

        % Damped least squares (Levenberg-Marquardt)
        lambda = 0.001;
        dphi = J' * ((J * J' + lambda * eye(3)) \ err);

        phi = phi + dphi;

        % Enforce joint limits
        limits = deg2rad([-170; -90; -90; -170]);
        limits_max = deg2rad([170; 90; 90; 170]);
        phi = max(limits, min(limits_max, phi));

        % Keep phi4 at gamma
        phi(4) = gamma;
    end
end

function J = numericalJacobian(phi)
%NUMERICALJACOBIAN 3x4 position Jacobian via finite differences
    delta = 1e-6;
    [p0, ~] = qarm_FK(phi);
    J = zeros(3, 4);
    for i = 1:4
        phi_d = phi;
        phi_d(i) = phi_d(i) + delta;
        [p_d, ~] = qarm_FK(phi_d);
        J(:, i) = (p_d - p0) / delta;
    end
end

function [phi_optimal, all_solutions] = analyticalIK(p, gamma)
%ANALYTICALIK Closed-form IK for QArm (4 solutions)
    px = p(1); py = p(2); pz = p(3);

    l1   = 0.14;
    l2   = sqrt(0.35^2 + 0.05^2);
    l3   = 0.25 + 0.15;
    beta = atan(0.05 / 0.35);

    A = l2; C = -l3; H = l1 - pz;
    D1 = sqrt(px^2 + py^2);
    D2 = -D1;

    theta2 = zeros(1,4);
    theta3 = zeros(1,4);

    for k = 1:2
        if k == 1, D = D1; else, D = D2; end
        F = (D^2 + H^2 - A^2 - C^2) / (2*A);
        disc = C^2 + F^2;
        if disc < 0, disc = 0; end
        t3a = 2 * atan2(-C + sqrt(disc), F);
        t3b = 2 * atan2(-C - sqrt(disc), F);
        for j = 1:2
            idx = (k-1)*2 + j;
            if j == 1, t3 = t3a; else, t3 = t3b; end
            theta3(idx) = t3;
            M = A + C*sin(t3);
            N = -C*cos(t3);
            denom = M^2 + N^2;
            if denom < 1e-12, denom = 1e-12; end
            ct2 = (D*M + H*N) / denom;
            st2 = (H - N*ct2) / M;
            theta2(idx) = atan2(st2, ct2);
        end
    end

    theta1 = zeros(1,4);
    for i = 1:4
        denom = l2*cos(theta2(i)) - l3*sin(theta2(i)+theta3(i));
        if abs(denom) < 1e-10
            theta1(i) = 0;
        else
            theta1(i) = atan2(py/denom, px/denom);
        end
    end

    theta4 = gamma * ones(1,4);

    phi = zeros(4,4);
    for i = 1:4
        phi(1,i) = wrapToPi(theta1(i));
        phi(2,i) = wrapToPi(theta2(i) + pi/2 - beta);
        phi(3,i) = wrapToPi(theta3(i) + beta);
        phi(4,i) = wrapToPi(theta4(i));
    end

    limits = deg2rad([-170 170; -90 90; -90 90; -170 170]);
    best_idx = 1; best_cost = inf;
    for i = 1:4
        valid = all(phi(:,i) >= limits(:,1)) && all(phi(:,i) <= limits(:,2));
        if valid
            cost = sum(phi(:,i).^2);
            if cost < best_cost
                best_cost = cost; best_idx = i;
            end
        end
    end

    phi_optimal = phi(:, best_idx);
    all_solutions = phi;
end
