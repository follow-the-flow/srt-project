"""
Forward and Inverse Kinematics for the Quanser QArm.
Pure Python/NumPy implementation (no MATLAB required).

QArm Physical Parameters:
  L1 = 0.14m  (base height)
  L2 = 0.35m  (upper arm)
  L3 = 0.05m  (elbow offset)
  L4 = 0.25m  (forearm)
  L5 = 0.15m  (wrist to end-effector)

DH Parameters (Standard Convention):
  | i | a_i    | alpha_i | d_i  | theta_i |
  | 1 | 0      | -pi/2   | l1   | theta_1 |
  | 2 | l2     | 0       | 0    | theta_2 |
  | 3 | 0      | -pi/2   | 0    | theta_3 |
  | 4 | 0      | 0       | l3   | theta_4 |
"""

import numpy as np

# Simplified link parameters
L1 = 0.14
L2 = 0.35
L3_OFFSET = 0.05
L4 = 0.25
L5 = 0.15

LAMBDA_1 = L1                                    # 0.14 m
LAMBDA_2 = np.sqrt(L2**2 + L3_OFFSET**2)         # 0.3536 m
LAMBDA_3 = L4 + L5                               # 0.40 m
BETA = np.arctan(L3_OFFSET / L2)                  # 0.1419 rad

# Joint limits (radians)
JOINT_LIMITS = np.array([
    [-2.967, 2.967],   # phi1: +/-170 deg
    [-1.571, 1.571],   # phi2: +/-90 deg
    [-1.571, 1.571],   # phi3: +/-90 deg
    [-2.967, 2.967],   # phi4: +/-170 deg
])


def dh_matrix(a, alpha, d, theta):
    """Compute the standard DH transformation matrix."""
    ct, st = np.cos(theta), np.sin(theta)
    ca, sa = np.cos(alpha), np.sin(alpha)
    return np.array([
        [ct, -st*ca,  st*sa, a*ct],
        [st,  ct*ca, -ct*sa, a*st],
        [0,   sa,     ca,    d   ],
        [0,   0,      0,     1   ]
    ])


def phi_to_theta(phi):
    """Convert physical joint angles (phi) to DH joint angles (theta)."""
    return np.array([
        phi[0],
        phi[1] - np.pi/2 + BETA,
        phi[2] - BETA,
        phi[3]
    ])


def theta_to_phi(theta):
    """Convert DH joint angles (theta) to physical joint angles (phi)."""
    return np.array([
        theta[0],
        theta[1] + np.pi/2 - BETA,
        theta[2] + BETA,
        theta[3]
    ])


def forward_kinematics(phi):
    """
    Compute forward kinematics for the QArm.

    Parameters
    ----------
    phi : array_like
        4-element array of physical joint angles [phi1, phi2, phi3, phi4] in radians.

    Returns
    -------
    p : np.ndarray
        3-element end-effector position [x, y, z] in metres.
    R : np.ndarray
        3x3 rotation matrix (end-effector orientation).
    """
    theta = phi_to_theta(np.asarray(phi, dtype=np.float64))

    T01 = dh_matrix(0,        -np.pi/2, LAMBDA_1, theta[0])
    T12 = dh_matrix(LAMBDA_2,  0,       0,        theta[1])
    T23 = dh_matrix(0,        -np.pi/2, 0,        theta[2])
    T34 = dh_matrix(0,         0,       LAMBDA_3, theta[3])

    T04 = T01 @ T12 @ T23 @ T34
    return T04[:3, 3].copy(), T04[:3, :3].copy()


def _analytical_ik(p, gamma=0.0):
    """
    Analytical IK: returns 4 solutions and the best valid one.
    """
    px, py, pz = p[0], p[1], p[2]
    l1, l2, l3 = LAMBDA_1, LAMBDA_2, LAMBDA_3

    A = l2
    C = -l3
    H = l1 - pz
    D1 = np.sqrt(px**2 + py**2)
    D2 = -D1

    theta2_all = np.zeros(4)
    theta3_all = np.zeros(4)

    for k in range(2):
        D = D1 if k == 0 else D2
        F = (D**2 + H**2 - A**2 - C**2) / (2*A)
        disc = C**2 + F**2
        if disc < 0:
            disc = 0
        sqrt_disc = np.sqrt(disc)
        t3a = 2 * np.arctan2(-C + sqrt_disc, F)
        t3b = 2 * np.arctan2(-C - sqrt_disc, F)

        for j in range(2):
            idx = k * 2 + j
            t3 = t3a if j == 0 else t3b
            theta3_all[idx] = t3
            M = A + C * np.sin(t3)
            N = -C * np.cos(t3)
            denom = M**2 + N**2
            if denom < 1e-12:
                denom = 1e-12
            ct2 = (D * M + H * N) / denom
            st2 = (H - N * ct2) / M if abs(M) > 1e-12 else 0
            theta2_all[idx] = np.arctan2(st2, ct2)

    theta1_all = np.zeros(4)
    for i in range(4):
        denom = l2 * np.cos(theta2_all[i]) - l3 * np.sin(theta2_all[i] + theta3_all[i])
        if abs(denom) < 1e-10:
            theta1_all[i] = 0
        else:
            theta1_all[i] = np.arctan2(py / denom, px / denom)

    theta4_all = np.full(4, gamma)

    # Convert to phi and wrap
    phi_all = np.zeros((4, 4))
    for i in range(4):
        phi_all[i] = theta_to_phi([theta1_all[i], theta2_all[i],
                                    theta3_all[i], theta4_all[i]])
        # Wrap to [-pi, pi]
        phi_all[i] = np.arctan2(np.sin(phi_all[i]), np.cos(phi_all[i]))

    # Select best valid solution
    best_idx = 0
    best_cost = np.inf
    for i in range(4):
        valid = np.all(phi_all[i] >= JOINT_LIMITS[:, 0]) and \
                np.all(phi_all[i] <= JOINT_LIMITS[:, 1])
        if valid:
            cost = np.sum(phi_all[i]**2)
            if cost < best_cost:
                best_cost = cost
                best_idx = i

    return phi_all[best_idx], phi_all


def _numerical_jacobian(phi):
    """Compute 3x4 position Jacobian via finite differences."""
    delta = 1e-6
    p0, _ = forward_kinematics(phi)
    J = np.zeros((3, 4))
    for i in range(4):
        phi_d = phi.copy()
        phi_d[i] += delta
        pd, _ = forward_kinematics(phi_d)
        J[:, i] = (pd - p0) / delta
    return J


def inverse_kinematics(p, gamma=0.0, max_iter=50, tol=1e-6):
    """
    Compute inverse kinematics with Newton-Raphson refinement.

    Parameters
    ----------
    p : array_like
        3-element target position [x, y, z] in metres.
    gamma : float
        Desired wrist angle (rad), typically 0.
    max_iter : int
        Maximum Newton-Raphson iterations.
    tol : float
        Convergence tolerance in metres.

    Returns
    -------
    phi : np.ndarray
        4-element optimal joint angles in radians.
    """
    p = np.asarray(p, dtype=np.float64)

    # Get analytical initial guess
    phi, _ = _analytical_ik(p, gamma)

    # Newton-Raphson refinement
    lam = 0.001  # Damping factor
    for _ in range(max_iter):
        p_curr, _ = forward_kinematics(phi)
        err = p - p_curr
        if np.linalg.norm(err) < tol:
            break

        J = _numerical_jacobian(phi)
        dphi = J.T @ np.linalg.solve(J @ J.T + lam * np.eye(3), err)
        phi = phi + dphi

        # Enforce limits
        phi = np.clip(phi, JOINT_LIMITS[:, 0], JOINT_LIMITS[:, 1])
        phi[3] = gamma

    return phi
