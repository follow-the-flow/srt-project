function [pos, segment_idx, seg_time] = multiSegmentTrajectory(waypoints, durations, t)
%MULTISEGMENTTRAJECTORY Navigate through multiple waypoints using cubic splines
%   INPUT:  waypoints  - 3xN matrix of waypoints
%           durations  - 1x(N-1) vector of segment durations (seconds)
%           t          - current time (seconds)
%   OUTPUT: pos         - 3x1 current position
%           segment_idx - which segment we're in (1-based)
%           seg_time    - time within current segment

    N = size(waypoints, 2);
    total_time = sum(durations);

    % Clamp time
    t = max(0, min(t, total_time));

    % Find which segment
    cumulative = 0;
    segment_idx = N - 1;  % Default to last
    seg_time = 0;
    for i = 1:(N-1)
        if t <= cumulative + durations(i)
            segment_idx = i;
            seg_time = t - cumulative;
            break;
        end
        cumulative = cumulative + durations(i);
    end

    % If past all segments, hold at final position
    if t >= total_time
        pos = waypoints(:, N);
        segment_idx = N - 1;
        seg_time = durations(end);
        return;
    end

    p_start = waypoints(:, segment_idx);
    p_end   = waypoints(:, segment_idx + 1);
    T_seg   = durations(segment_idx);

    pos = cubicTrajectory(p_start, p_end, T_seg, seg_time);
end
