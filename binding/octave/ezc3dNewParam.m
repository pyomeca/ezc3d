function out = ezc3dNewParam(data, description, is_locked)
    if nargin < 3
        is_locked = false;
    end
    if nargin < 2
        description = '';
    end
    if nargin < 1
        data = [];
    end
    
    out = struct('DESCRIPTION', description, 'IS_LOCKED', is_locked, 'DATA', []);
    if iscell(data)
        if size(data, 1) == 1
            data = data';
        end
        
        if size(data, 2) ~= 1
            error('Parameters from a cell must a vector')
        end
        out.DATA = data;
    elseif ischar(data)
        out.DATA = {data};
    else
        out.DATA = data;
    end
end
