clc
clear
close all
fprintf('Matlab tests started..\n\n')

% Load an empty c3d structure
fprintf('Loading an empty C3D.. ');
c3d = ezc3dRead();

% Test the header of the new C3D
assert(c3d.header.points.size == 0)
assert(c3d.header.points.frameRate == 0)
assert(c3d.header.points.firstFrame == 1)
assert(c3d.header.points.lastFrame == 1)

assert(c3d.header.analogs.size == 0)
assert(c3d.header.analogs.frameRate == 0)
assert(c3d.header.analogs.firstFrame == 1)
assert(c3d.header.analogs.lastFrame == 0)

assert(c3d.header.events.size == 18)
for i = 1:18
    assert(c3d.header.events.eventsTime(i) == 0)
    assert(isempty(c3d.header.events.eventsLabel{i}))
    assert(ischar(c3d.header.events.eventsLabel{i}))
end

% Test the parameters
assert(c3d.parameters.POINT.USED == 0)
assert(c3d.parameters.POINT.SCALE == -1)
assert(c3d.parameters.POINT.RATE == 0)
assert(c3d.parameters.POINT.FRAMES == 0)
assert(isempty(c3d.parameters.POINT.LABELS))
assert(isempty(c3d.parameters.POINT.DESCRIPTIONS))
assert(isempty(c3d.parameters.POINT.UNITS))

assert(c3d.parameters.ANALOG.USED == 0)
assert(isempty(c3d.parameters.ANALOG.LABELS))
assert(isempty(c3d.parameters.ANALOG.DESCRIPTIONS))
assert(c3d.parameters.ANALOG.GEN_SCALE == 1)
assert(isempty(c3d.parameters.ANALOG.SCALE))
assert(isempty(c3d.parameters.ANALOG.OFFSET))
assert(isempty(c3d.parameters.ANALOG.UNITS))
assert(c3d.parameters.ANALOG.RATE == 0)
assert(isempty(c3d.parameters.ANALOG.FORMAT))
assert(isempty(c3d.parameters.ANALOG.BITS))

assert(c3d.parameters.FORCE_PLATFORM.USED == 0)
assert(isempty(c3d.parameters.FORCE_PLATFORM.TYPE))
assert(size(c3d.parameters.FORCE_PLATFORM.ZERO, 1) == 2)
FORCE_PLATFORM.ZERO = [1, 0];
for i=1:length(c3d.parameters.FORCE_PLATFORM.ZERO)
    assert(c3d.parameters.FORCE_PLATFORM.ZERO(i) == FORCE_PLATFORM.ZERO(i))
end
assert(isempty(c3d.parameters.FORCE_PLATFORM.CORNERS))
assert(isempty(c3d.parameters.FORCE_PLATFORM.ORIGIN))
assert(isempty(c3d.parameters.FORCE_PLATFORM.CHANNEL))
assert(isempty(c3d.parameters.FORCE_PLATFORM.CAL_MATRIX))

% Test the data
assert(isempty(c3d.data.points))
assert(sum(size(c3d.data.points) == [3, 0, 0]) == 3)
assert(isempty(c3d.data.analogs))
assert(sum(size(c3d.data.analogs) == [0, 0]) == 2)
fprintf('Done\n');

% Now add some data.
fprintf('Writing and reading back a C3D.. ');

% Adjust some mandatory values of the parameters and fill 
% the data with random values
nbSeconds = 2;
pointFrameRate = 100;
pointNames = {'point1', 'point2', 'point3', 'point4', 'point5'};
pointData = rand(3, length(pointNames), pointFrameRate * nbSeconds);
analogFrameRate = 1000;
analogNames = {'analog1', 'analog2', 'analog3', ...
               'analog4', 'analog5', 'analog6'};
analogData = rand(analogFrameRate * nbSeconds, length(analogNames));
           


c3d.parameters.POINT.RATE = pointFrameRate;
c3d.parameters.POINT.LABELS = pointNames;
c3d.data.points = pointData;

c3d.parameters.ANALOG.RATE = analogFrameRate;
c3d.parameters.ANALOG.LABELS = analogNames;
c3d.data.analogs = analogData;
 
% Create a custom parameter to the POINT group
newPointParam = {'POINT', 'newParam', [1, 2, 3]};
c3d.parameters.(newPointParam{1}).(newPointParam{2}) = newPointParam{3};

% Create a custom parameter a new group
newGroupParam = {'newGroup', 'newParam', {'MyParam1', 'MyParam2'}};
c3d.parameters.(newGroupParam{1}).(newGroupParam{2}) = newGroupParam{3};
 
% Write a new modified C3D and read back the data
ezc3dWrite('temporary.c3d', c3d);
c3d_to_compare = ezc3dRead('temporary.c3d');

% Test the opened file
% Test the header
assert(c3d_to_compare.header.points.size == length(pointNames))
assert(c3d_to_compare.header.points.frameRate == pointFrameRate)
assert(c3d_to_compare.header.points.firstFrame == 1)
assert(c3d_to_compare.header.points.lastFrame == pointFrameRate * nbSeconds)

assert(c3d_to_compare.header.analogs.size == length(analogNames))
assert(c3d_to_compare.header.analogs.frameRate == analogFrameRate)
assert(c3d_to_compare.header.analogs.firstFrame == 1)
assert(c3d_to_compare.header.analogs.lastFrame == analogFrameRate * nbSeconds)

assert(c3d_to_compare.header.events.size == 18)
for i = 1:18
    assert(c3d_to_compare.header.events.eventsTime(i) == 0)
    assert(isempty(c3d_to_compare.header.events.eventsLabel{i}))
    assert(ischar(c3d_to_compare.header.events.eventsLabel{i}))
end

% Test the parameters
assert(c3d_to_compare.parameters.POINT.USED == length(pointNames))
assert(c3d_to_compare.parameters.POINT.SCALE == -1)
assert(c3d_to_compare.parameters.POINT.RATE == pointFrameRate)
assert(c3d_to_compare.parameters.POINT.FRAMES == pointFrameRate * nbSeconds)
assert(length(c3d_to_compare.parameters.POINT.LABELS) == length(pointNames))
assert(length(c3d_to_compare.parameters.POINT.DESCRIPTIONS) == length(pointNames))
for i = 1:length(pointNames)
    assert(strcmp(c3d_to_compare.parameters.POINT.LABELS{i}, pointNames{i}))
    assert(isempty(c3d_to_compare.parameters.POINT.DESCRIPTIONS{i}))
end
assert(isempty(c3d_to_compare.parameters.POINT.UNITS))
assert(length(c3d_to_compare.parameters.(newPointParam{1}).(upper(newPointParam{2}))) == length(newPointParam{3}))
for i = 1:length(newPointParam{3})
    assert(c3d_to_compare.parameters.(newPointParam{1}).(upper(newPointParam{2}))(i) == newPointParam{3}(i))
end

assert(c3d_to_compare.parameters.ANALOG.USED == length(analogNames))
assert(c3d_to_compare.parameters.ANALOG.RATE == analogFrameRate)
assert(length(c3d_to_compare.parameters.ANALOG.LABELS) == length(analogNames))
assert(length(c3d_to_compare.parameters.ANALOG.DESCRIPTIONS) == length(analogNames))
assert(length(c3d_to_compare.parameters.ANALOG.SCALE) == length(analogNames))
assert(length(c3d_to_compare.parameters.ANALOG.OFFSET) == length(analogNames))
assert(length(c3d_to_compare.parameters.ANALOG.UNITS) == length(analogNames))
for i = 1:length(analogNames)
    assert(strcmp(c3d_to_compare.parameters.ANALOG.LABELS{i}, analogNames{i}))
    assert(isempty(c3d_to_compare.parameters.ANALOG.DESCRIPTIONS{i}))
    assert(c3d_to_compare.parameters.ANALOG.SCALE(i) == 1)
    assert(c3d_to_compare.parameters.ANALOG.OFFSET(i) == 0)
    assert(strcmp(c3d_to_compare.parameters.ANALOG.UNITS{i}, 'V'))
end
assert(c3d_to_compare.parameters.ANALOG.GEN_SCALE == 1)
assert(isempty(c3d_to_compare.parameters.ANALOG.FORMAT))
assert(isempty(c3d_to_compare.parameters.ANALOG.BITS))

assert(c3d_to_compare.parameters.FORCE_PLATFORM.USED == 0)
assert(isempty(c3d_to_compare.parameters.FORCE_PLATFORM.TYPE))
assert(size(c3d_to_compare.parameters.FORCE_PLATFORM.ZERO, 1) == 2)
FORCE_PLATFORM.ZERO = [1, 0];
for i=1:length(c3d_to_compare.parameters.FORCE_PLATFORM.ZERO)
    assert(c3d_to_compare.parameters.FORCE_PLATFORM.ZERO(i) == FORCE_PLATFORM.ZERO(i))
end
assert(isempty(c3d_to_compare.parameters.FORCE_PLATFORM.CORNERS))
assert(isempty(c3d_to_compare.parameters.FORCE_PLATFORM.ORIGIN))
assert(isempty(c3d_to_compare.parameters.FORCE_PLATFORM.CHANNEL))
assert(isempty(c3d_to_compare.parameters.FORCE_PLATFORM.CAL_MATRIX))

assert(length(c3d_to_compare.parameters.(upper(newGroupParam{1})).(upper(newGroupParam{2}))) == length(newGroupParam{3}))
for i = 1:length(newGroupParam{3})
    assert(strcmp(c3d_to_compare.parameters.(upper(newGroupParam{1})).(upper(newGroupParam{2})){i}, newGroupParam{3}{i}))
end

% Test the data
assert(sum(size(c3d_to_compare.data.points) == [3, length(pointNames),  pointFrameRate * nbSeconds]) == 3)
assert(sum(sum(sum(abs(c3d_to_compare.data.points - pointData) < 1e-6))) == 3 * length(pointNames) *  pointFrameRate * nbSeconds)
assert(sum(size(c3d_to_compare.data.analogs) == [analogFrameRate * nbSeconds, length(analogNames)]) == 2)
assert(sum(sum(abs(c3d_to_compare.data.analogs - analogData) < 1e-6)) == analogFrameRate * nbSeconds * length(analogNames))
fprintf('Done\n');

fprintf('\nMatlab tests successfully completed..\n')