function phi = py_ik(p_target)
%PY_IK Inverse kinematics via the Python qarm_kinematics module.
%   Wraps py.qarm_kinematics.inverse_kinematics so a Simulink MATLAB Function
%   block can call it through coder.extrinsic. Input: 3x1 Cartesian target (m).
%   Output: 4x1 joint angles (rad).

    coder.extrinsic('py.numpy.array', 'py.importlib.import_module', 'double');
    phi = zeros(4,1);

    k = py.importlib.import_module('qarm_kinematics');
    np_p = py.numpy.array({p_target(1), p_target(2), p_target(3)});
    phi_py = k.inverse_kinematics(np_p);
    phi = double(phi_py);
    phi = phi(:);
end
