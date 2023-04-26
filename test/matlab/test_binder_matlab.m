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
assert(c3d.parameters.POINT.USED.DATA == 0)
assert(c3d.parameters.POINT.SCALE.DATA == -1)
assert(c3d.parameters.POINT.RATE.DATA == 0)
assert(c3d.parameters.POINT.FRAMES.DATA == 0)
assert(isempty(c3d.parameters.POINT.LABELS.DATA))
assert(isempty(c3d.parameters.POINT.DESCRIPTIONS.DATA))
assert(isempty(c3d.parameters.POINT.UNITS.DATA))

assert(c3d.parameters.ANALOG.USED.DATA == 0)
assert(isempty(c3d.parameters.ANALOG.LABELS.DATA))
assert(isempty(c3d.parameters.ANALOG.DESCRIPTIONS.DATA))
assert(c3d.parameters.ANALOG.GEN_SCALE.DATA == 1)
assert(isempty(c3d.parameters.ANALOG.SCALE.DATA))
assert(isempty(c3d.parameters.ANALOG.OFFSET.DATA))
assert(isempty(c3d.parameters.ANALOG.UNITS.DATA))
assert(c3d.parameters.ANALOG.RATE.DATA == 0)
assert(isempty(c3d.parameters.ANALOG.FORMAT.DATA))
assert(isempty(c3d.parameters.ANALOG.BITS.DATA))

assert(c3d.parameters.FORCE_PLATFORM.USED.DATA == 0)
assert(isempty(c3d.parameters.FORCE_PLATFORM.TYPE.DATA))
assert(size(c3d.parameters.FORCE_PLATFORM.ZERO.DATA, 1) == 2)
FORCE_PLATFORM.ZERO = [1, 0];
for i=1:length(c3d.parameters.FORCE_PLATFORM.ZERO)
    assert(c3d.parameters.FORCE_PLATFORM.ZERO.DATA(i) == FORCE_PLATFORM.ZERO(i))
end
assert(isempty(c3d.parameters.FORCE_PLATFORM.CORNERS.DATA))
assert(isempty(c3d.parameters.FORCE_PLATFORM.ORIGIN.DATA))
assert(isempty(c3d.parameters.FORCE_PLATFORM.CHANNEL.DATA))
assert(isempty(c3d.parameters.FORCE_PLATFORM.CAL_MATRIX.DATA))

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
           


c3d.parameters.POINT.RATE.DATA = pointFrameRate;
c3d.parameters.POINT.LABELS.DATA = pointNames;
c3d.data.points = pointData;

c3d.parameters.ANALOG.RATE.DATA = analogFrameRate;
c3d.parameters.ANALOG.LABELS.DATA = analogNames;
c3d.data.analogs = analogData;
 
% Create a custom parameter to the POINT group
newPointParam = {'POINT', 'newParam', [1, 2, 3]};
c3d.parameters.(newPointParam{1}).(newPointParam{2}) = ezc3dNewParam(newPointParam{3});

% Create a custom parameter a new group
newGroupParam = {'newGroup', 'newParam', {'MyParam1', 'MyParam2'}};
c3d.parameters.(newGroupParam{1}).(newGroupParam{2}) = ezc3dNewParam(newGroupParam{3});
 
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
assert(c3d_to_compare.parameters.POINT.USED.DATA == length(pointNames))
assert(c3d_to_compare.parameters.POINT.SCALE.DATA == -1)
assert(c3d_to_compare.parameters.POINT.RATE.DATA == pointFrameRate)
assert(c3d_to_compare.parameters.POINT.FRAMES.DATA == pointFrameRate * nbSeconds)
assert(length(c3d_to_compare.parameters.POINT.LABELS.DATA) == length(pointNames))
assert(length(c3d_to_compare.parameters.POINT.DESCRIPTIONS.DATA) == length(pointNames))
for i = 1:length(pointNames)
    assert(strcmp(c3d_to_compare.parameters.POINT.LABELS.DATA{i}, pointNames{i}))
    assert(isempty(c3d_to_compare.parameters.POINT.DESCRIPTIONS.DATA{i}))
end
assert(isempty(c3d_to_compare.parameters.POINT.UNITS.DATA))
assert(length(c3d_to_compare.parameters.(newPointParam{1}).(newPointParam{2}).DATA) == length(newPointParam{3}))
for i = 1:length(newPointParam{3})
    assert(c3d_to_compare.parameters.(newPointParam{1}).(newPointParam{2}).DATA(i) == newPointParam{3}(i))
end

assert(c3d_to_compare.parameters.ANALOG.USED.DATA == length(analogNames))
assert(c3d_to_compare.parameters.ANALOG.RATE.DATA == analogFrameRate)
assert(length(c3d_to_compare.parameters.ANALOG.LABELS.DATA) == length(analogNames))
assert(length(c3d_to_compare.parameters.ANALOG.DESCRIPTIONS.DATA) == length(analogNames))
assert(length(c3d_to_compare.parameters.ANALOG.SCALE.DATA) == length(analogNames))
assert(length(c3d_to_compare.parameters.ANALOG.OFFSET.DATA) == length(analogNames))
assert(length(c3d_to_compare.parameters.ANALOG.UNITS.DATA) == length(analogNames))
for i = 1:length(analogNames)
    assert(strcmp(c3d_to_compare.parameters.ANALOG.LABELS.DATA{i}, analogNames{i}))
    assert(isempty(c3d_to_compare.parameters.ANALOG.DESCRIPTIONS.DATA{i}))
    assert(c3d_to_compare.parameters.ANALOG.SCALE.DATA(i) == 1)
    assert(c3d_to_compare.parameters.ANALOG.OFFSET.DATA(i) == 0)
    assert(strcmp(c3d_to_compare.parameters.ANALOG.UNITS.DATA{i}, ''))
end
assert(c3d_to_compare.parameters.ANALOG.GEN_SCALE.DATA == 1)
assert(isempty(c3d_to_compare.parameters.ANALOG.FORMAT.DATA))
assert(isempty(c3d_to_compare.parameters.ANALOG.BITS.DATA))

assert(c3d_to_compare.parameters.FORCE_PLATFORM.USED.DATA == 0)
assert(isempty(c3d_to_compare.parameters.FORCE_PLATFORM.TYPE.DATA))
assert(size(c3d_to_compare.parameters.FORCE_PLATFORM.ZERO.DATA, 1) == 2)
FORCE_PLATFORM.ZERO = [1, 0];
for i=1:length(c3d_to_compare.parameters.FORCE_PLATFORM.ZERO)
    assert(c3d_to_compare.parameters.FORCE_PLATFORM.ZERO.DATA(i) == FORCE_PLATFORM.ZERO(i))
end
assert(isempty(c3d_to_compare.parameters.FORCE_PLATFORM.CORNERS.DATA))
assert(isempty(c3d_to_compare.parameters.FORCE_PLATFORM.ORIGIN.DATA))
assert(isempty(c3d_to_compare.parameters.FORCE_PLATFORM.CHANNEL.DATA))
assert(isempty(c3d_to_compare.parameters.FORCE_PLATFORM.CAL_MATRIX.DATA))

assert(length(c3d_to_compare.parameters.(newGroupParam{1}).(newGroupParam{2}).DATA) == length(newGroupParam{3}))
for i = 1:length(newGroupParam{3})
    assert(strcmp(c3d_to_compare.parameters.(newGroupParam{1}).(newGroupParam{2}).DATA{i}, newGroupParam{3}{i}))
end

% Test the data
assert(sum(size(c3d_to_compare.data.points) == [3, length(pointNames),  pointFrameRate * nbSeconds]) == 3)
assert(sum(sum(sum(abs(c3d_to_compare.data.points - pointData) < 1e-6))) == 3 * length(pointNames) *  pointFrameRate * nbSeconds)
assert(sum(size(c3d_to_compare.data.analogs) == [analogFrameRate * nbSeconds, length(analogNames)]) == 2)
assert(sum(sum(abs(c3d_to_compare.data.analogs - analogData) < 1e-6)) == analogFrameRate * nbSeconds * length(analogNames))
fprintf('Done\n');

% Test the event adder from a file which does not have event yet
c3d = ezc3dRead('../c3dFiles/ezc3d-testFiles-master/ezc3d-testFiles-master/Optotrak.c3d');
c3d = ezc3dAddEvent(c3d, [0, 0.1], 'Left', 'MyNewEvent', 'Hey! This is new!', 'Me', 2, 1);
% Only one event
ezc3dWrite('temporary.c3d', c3d);
c3dToCompare = ezc3dRead('temporary.c3d');
assert(c3dToCompare.parameters.EVENT.USED.DATA == 1)
assert(sum(sum(abs(c3dToCompare.parameters.EVENT.TIMES.DATA - [0; 0.1]))) < 1e-5);
assert(strcmp(c3dToCompare.parameters.EVENT.CONTEXTS.DATA{1}, 'Left'));
assert(strcmp(c3dToCompare.parameters.EVENT.LABELS.DATA{1}, 'MyNewEvent'));
assert(strcmp(c3dToCompare.parameters.EVENT.DESCRIPTIONS.DATA{1}, 'Hey! This is new!'));
assert(strcmp(c3dToCompare.parameters.EVENT.SUBJECTS.DATA{1}, 'Me'));
assert(sum(c3dToCompare.parameters.EVENT.ICON_IDS.DATA - 2) == 0);
assert(sum(c3dToCompare.parameters.EVENT.GENERIC_FLAGS.DATA - 1) == 0);

% More than one
c3d = ezc3dAddEvent(c3d, [0, 0.2]);
ezc3dWrite('temporary.c3d', c3d);
c3dToCompare = ezc3dRead('temporary.c3d');
assert(c3dToCompare.parameters.EVENT.USED.DATA == 2)
assert(sum(sum(abs(c3dToCompare.parameters.EVENT.TIMES.DATA - [0 0; 0.1 0.2]))) < 1e-5);
assert(strcmp(c3dToCompare.parameters.EVENT.CONTEXTS.DATA{1}, 'Left'));
assert(strcmp(c3dToCompare.parameters.EVENT.CONTEXTS.DATA{2}, ''));
assert(strcmp(c3dToCompare.parameters.EVENT.LABELS.DATA{1}, 'MyNewEvent'));
assert(strcmp(c3dToCompare.parameters.EVENT.LABELS.DATA{2}, ''));
assert(strcmp(c3dToCompare.parameters.EVENT.DESCRIPTIONS.DATA{1}, 'Hey! This is new!'));
assert(strcmp(c3dToCompare.parameters.EVENT.DESCRIPTIONS.DATA{2}, ''));
assert(strcmp(c3dToCompare.parameters.EVENT.SUBJECTS.DATA{1}, 'Me'));
assert(strcmp(c3dToCompare.parameters.EVENT.SUBJECTS.DATA{2}, ''));
assert(sum(c3dToCompare.parameters.EVENT.ICON_IDS.DATA - [2, 0]) == 0);
assert(sum(c3dToCompare.parameters.EVENT.GENERIC_FLAGS.DATA - [1, 0]) == 0);

% Add from a previously loaded file
c3d = c3dToCompare;
c3d = ezc3dAddEvent(c3d, [0, 0.3], 'Right', 'MySecondNewEvent', 'Hey! This is new again!', 'You', 3, 2);
c3d = ezc3dAddEvent(c3d, [0, 0.4]);
ezc3dWrite('temporary.c3d', c3d);
c3dToCompare = ezc3dRead('temporary.c3d');
assert(c3dToCompare.parameters.EVENT.USED.DATA == 4)
assert(sum(sum(abs(c3dToCompare.parameters.EVENT.TIMES.DATA - [0 0 0 0; 0.1 0.2 0.3 0.4]))) < 1e-5);
strcmp(c3dToCompare.parameters.EVENT.CONTEXTS.DATA{1}, 'Left');
strcmp(c3dToCompare.parameters.EVENT.CONTEXTS.DATA{2}, '');
strcmp(c3dToCompare.parameters.EVENT.CONTEXTS.DATA{3}, 'Right');
strcmp(c3dToCompare.parameters.EVENT.CONTEXTS.DATA{4}, '');
strcmp(c3dToCompare.parameters.EVENT.LABELS.DATA{1}, 'MyNewEvent');
strcmp(c3dToCompare.parameters.EVENT.LABELS.DATA{2}, '');
strcmp(c3dToCompare.parameters.EVENT.LABELS.DATA{3}, 'MySecondNewEvent');
strcmp(c3dToCompare.parameters.EVENT.LABELS.DATA{4}, '');
strcmp(c3dToCompare.parameters.EVENT.DESCRIPTIONS.DATA{1}, 'Hey! This is new!');
strcmp(c3dToCompare.parameters.EVENT.DESCRIPTIONS.DATA{2}, '');
strcmp(c3dToCompare.parameters.EVENT.DESCRIPTIONS.DATA{3}, 'Hey! This is new again!');
strcmp(c3dToCompare.parameters.EVENT.DESCRIPTIONS.DATA{4}, '');
strcmp(c3dToCompare.parameters.EVENT.SUBJECTS.DATA{1}, 'Me');
strcmp(c3dToCompare.parameters.EVENT.SUBJECTS.DATA{2}, '');
strcmp(c3dToCompare.parameters.EVENT.SUBJECTS.DATA{3}, 'Me');
strcmp(c3dToCompare.parameters.EVENT.SUBJECTS.DATA{4}, '');
assert(sum(c3dToCompare.parameters.EVENT.ICON_IDS.DATA - [2, 0, 3, 0]) == 0);
assert(sum(c3dToCompare.parameters.EVENT.GENERIC_FLAGS.DATA - [1, 0, 2, 0]) == 0);

% Now test for a file with Rotation
c3d = ezc3dRead('../c3dFiles/ezc3d-testFiles-master/ezc3d-testFiles-master/C3DRotationExample.c3d');
assert(sum(size(c3d.data.rotations) == [4, 4, 21, 340]) == 4)
comparisonValue = [-0.4584634900  0.8721544743 -0.1707565784 -455.0068664551
     -0.3688277900 -0.3615344167 -0.8563054204 1048.7111816406
     -0.8085649610 -0.3296049833  0.4874251187  991.1699218750
      0.0000000000  0.0000000000  0.0000000000    1.0000000000];
assert(sum(sum(c3d.data.rotations(:, :, 15, 256) - comparisonValue)) < 1e-8);

ezc3dWrite('temporary.c3d', c3d);
c3dToCompare = ezc3dRead('temporary.c3d');
assert(sum(size(c3dToCompare.data.rotations) == [4, 4, 21, 340]) == 4)
assert(sum(sum(c3dToCompare.data.rotations(:, :, 15, 256) - comparisonValue)) < 1e-8);

% All done!
delete('temporary.c3d');
fprintf('\nMatlab tests successfully completed!\n')

