function pos = cubicTrajectory(p_start, p_end, T, t)
%CUBICTRAJECTORY Evaluates cubic spline trajectory at time t
%   Zero velocity at start and end (rest-to-rest motion)
%   INPUT:  p_start - 3x1 start position
%           p_end   - 3x1 end position
%           T       - segment duration (seconds)
%           t       - current time within segment [0, T]
%   OUTPUT: pos     - 3x1 interpolated position

    t = max(0, min(t, T));  % Clamp to [0, T]

    a0 = p_start;
    a2 = 3*(p_end - p_start) / T^2;
    a3 = -2*(p_end - p_start) / T^3;

    pos = a0 + a2*t^2 + a3*t^3;
end
