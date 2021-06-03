function c3d = ezc3dAddEvent(c3d, time, context, label, description, subject, icon_id, generic_flag)
% This function adds an event, warning two events can have the same name (it wont't override it)
% 
% c3d: The C3D struct to add the event to
% time: A 2 element vector for the time, first element is the time in minute (integer), second is the second (float)
% context: The context (usually 'Right', 'Left' or 'General')
% label: The name of the event
% description: The description of the event
% subject: The subject the event is applied to. An empty string is generic or the only subject in the scene
% icon_id: The ID of the icon of the event
% generic_flag: A generic flag

    if length(time) ~= 2
        error('time should be a 2 element vector (minutes as integer, seconds as double)');
    end
    if size(time, 1) == 1
        time = time';
    end
    
    if nargin < 3
        context = '';
    end
    if nargin < 4
        label = '';
    end
    if nargin < 5
        description = '';
    end
    if nargin < 6
        subject = '';
    end
    if nargin < 7
        icon_id = 0;
    end
    if nargin < 8
        generic_flag = 0;
    end
    
    % Create the EVENT group if it does not exist
    if ~isfield(c3d.parameters, 'EVENT')
        c3d.parameters.EVENT = struct(...
            'USED', ezc3dNewParam(1), ...
            'TIMES', ezc3dNewParam(time), ...
            'CONTEXTS', ezc3dNewParam(context), ...
            'LABELS', ezc3dNewParam(label), ...
            'DESCRIPTIONS', ezc3dNewParam(description), ...
            'SUBJECTS', ezc3dNewParam(subject), ...
            'ICON_IDS', ezc3dNewParam(icon_id), ...
            'GENERIC_FLAGS', ezc3dNewParam(generic_flag) ...
        );
    else
        c3d.parameters.EVENT.USED.DATA = c3d.parameters.EVENT.USED.DATA + 1;
        c3d.parameters.EVENT.TIMES.DATA = [c3d.parameters.EVENT.TIMES.DATA, time];
        c3d.parameters.EVENT.CONTEXTS.DATA = [c3d.parameters.EVENT.CONTEXTS.DATA, {context}];
        c3d.parameters.EVENT.LABELS.DATA = [c3d.parameters.EVENT.LABELS.DATA, {label}];
        c3d.parameters.EVENT.DESCRIPTIONS.DATA = [c3d.parameters.EVENT.DESCRIPTIONS.DATA, {description}];
        c3d.parameters.EVENT.SUBJECTS.DATA = [c3d.parameters.EVENT.SUBJECTS.DATA, {subject}];
        c3d.parameters.EVENT.ICON_IDS.DATA = [c3d.parameters.EVENT.ICON_IDS.DATA, icon_id];
        c3d.parameters.EVENT.GENERIC_FLAGS.DATA = [c3d.parameters.EVENT.GENERIC_FLAGS.DATA, generic_flag];
    
    end  
end
