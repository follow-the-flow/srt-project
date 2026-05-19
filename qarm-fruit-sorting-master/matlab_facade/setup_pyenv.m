function setup_pyenv()
%SETUP_PYENV Configure MATLAB to use the project's Python interpreter.
%   Called by every launcher/build script so the py.* bridge is warm and the
%   project's python/ folder is on sys.path before any block executes.

    pe = pyenv;
    if pe.Status ~= "Loaded"
        pyenv('Version', 'C:\Python313\python.exe', ...
              'ExecutionMode', 'InProcess');
    end

    facade_dir = fileparts(mfilename('fullpath'));
    project_dir = fileparts(facade_dir);
    python_dir = fullfile(project_dir, 'python');

    sys_path = py.sys.path;
    if count(sys_path, python_dir) == 0
        sys_path.insert(int32(0), python_dir);
    end
end
