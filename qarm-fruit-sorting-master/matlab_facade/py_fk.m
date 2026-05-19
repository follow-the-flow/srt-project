function p = py_fk(phi)
%PY_FK Forward kinematics via the Python qarm_kinematics module.
%   Returns end-effector position [x;y;z] in metres for joint vector phi (rad).

    coder.extrinsic('py.numpy.array', 'py.importlib.import_module', 'cell', 'double');
    p = zeros(3,1);

    k = py.importlib.import_module('qarm_kinematics');
    np_phi = py.numpy.array({phi(1), phi(2), phi(3), phi(4)});
    tup = k.forward_kinematics(np_phi);
    c = cell(tup);
    p = double(c{1});
    p = p(:);
end
