function py_remote_view(action)
%PY_REMOTE_VIEW Launch or close the OpenCV companion window for remote mode.
%   py_remote_view('open')   -- start the Python window (non-blocking).
%   py_remote_view('close')  -- close it.
%   Calling with no arg defaults to 'open'. The window runs in a
%   separate Python process so it does not block Simulink.
%
%   Uses MATLAB's system() with '&' (Windows 'start') to launch detached.
%   We track the PID in a persistent so 'close' can kill it.

    if nargin < 1, action = 'open'; end
    persistent pid
    pyexe = 'C:/Python313/python.exe';
    script = fullfile(fileparts(mfilename('fullpath')), '..', 'python', ...
                       'remote_view.py');
    switch action
        case 'open'
            if isempty(pid) || ~isPidAlive(pid)
                cmd = sprintf('start "remote_view" /MIN "%s" "%s"', pyexe, script);
                [status, out] = system(cmd);
                if status ~= 0
                    warning('py_remote_view: launch failed (status=%d): %s', status, out);
                    return;
                end
                % Best-effort PID capture via tasklist; if this fails the
                % user can kill python.exe manually. Not a blocker.
                [~, list] = system('tasklist /FI "IMAGENAME eq python.exe" /NH /FO CSV');
                toks = regexp(list, '"python.exe","(\d+)"', 'tokens');
                if ~isempty(toks)
                    pid = str2double(toks{end}{1});
                else
                    pid = 0;
                end
            end
        case 'close'
            if ~isempty(pid) && pid > 0 && isPidAlive(pid)
                system(sprintf('taskkill /PID %d /F', pid));
                pid = [];
            end
        otherwise
            error('py_remote_view: unknown action "%s"', action);
    end
end


function alive = isPidAlive(pid)
    if isempty(pid) || pid <= 0
        alive = false; return;
    end
    [~, out] = system(sprintf('tasklist /FI "PID eq %d" /NH', pid));
    alive = ~isempty(regexp(out, sprintf('\\s%d\\s', pid), 'once'));
end
