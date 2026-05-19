function [phi, gripper, state_id, done] = py_controller( ...
                                joints_cur, dt, reset, fruit_list)
%PY_CONTROLLER Persistent handle to py.sorting_controller_sim.StepController.
%   Maintains the Python FSM instance across Simulink solver steps. Pass
%   reset=1 once (e.g. on the first tick) to (re)create the controller.
%
%   Inputs
%   ------
%   joints_cur : 4x1 double, current joint angles (rad)
%   dt         : scalar double, solver step (s)
%   reset      : scalar, non-zero rebuilds the controller
%   fruit_list : optional N x 5 double [type_id, wx, wy, wz, conf] — output
%                of py_detect_live. When empty or omitted, the controller
%                is (re)built with a hardcoded demo queue so the FSM slice
%                can simulate without vision in the loop.
%                type_id: 1 strawberry, 2 tomato, 3 banana. Rows with
%                type_id==0 are treated as empty and ignored.
%
%   Outputs
%   -------
%   phi      : 4x1 double, joint command (rad)
%   gripper  : scalar, 0=open, 1=closed
%   state_id : int32, FSM state
%   done     : int32, 1 when sorting finished

    persistent ctrl

    if nargin < 4
        fruit_list = [];
    end

    if isempty(ctrl) || reset ~= 0
        [positions, types] = build_queue(fruit_list);
        ctrl = py.sorting_controller_sim.StepController(positions, types);
    end

    j = py.numpy.array(joints_cur(:)');
    result = ctrl.step(j, dt);
    phi_py    = result{1};
    gripper   = double(result{2});
    state_id  = int32(result{3});
    done      = int32(result{4});
    phi = double(phi_py)';
    phi = phi(:);
end


function [positions, types] = build_queue(fruit_list)
%BUILD_QUEUE Convert an N x 5 fruit matrix to py.list inputs for the
%StepController constructor. When fruit_list is empty, falls back to the
%hardcoded demo queue used by the FSM vertical-slice tests.

    type_name = {'strawberry', 'tomato', 'banana'};

    if isempty(fruit_list)
        % Empty queue = nothing to sort. Autonomous mode drives via
        % main_final.py shell-out (see spec §4.4), not via Simulink FSM.
        positions = py.list({});
        types = py.list({});
        return
    end

    positions = py.list({});
    types = py.list({});
    for i = 1:size(fruit_list, 1)
        tid = int32(fruit_list(i, 1));
        if tid < 1 || tid > 3
            continue  % 0 = empty row, anything else ignored
        end
        wx = fruit_list(i, 2);
        wy = fruit_list(i, 3);
        wz = fruit_list(i, 4);
        positions.append(py.list({wx, wy, wz}));
        types.append(type_name{tid});
    end
end
