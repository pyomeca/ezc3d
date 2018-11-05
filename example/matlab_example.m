clc
clear
close all
% addpath to ezc3d here

% Load an empty c3d structure
c3d = ezc3dRead();

% Adjust some mandatory values of the parameters and fill 
% the data with random values
c3d.parameters.POINT.RATE = 100;
c3d.parameters.POINT.LABELS = {'point1', 'point2', 'point3', ...
                               'point4', 'point5'};
c3d.data.points = rand(3, 5, 100);

c3d.parameters.ANALOG.RATE = 1000;
c3d.parameters.ANALOG.LABELS = {'analog1', 'analog2', 'analog3', ...
                                'analog4', 'analog5', 'analog6'};
c3d.data.analogs = rand(1000, 6);
 
% Create a custom parameter to the POINT group
c3d.parameters.POINT.newParam = [1, 2, 3];

% Create a custom parameter a new group
c3d.parameters.NewGroup.newParam = {'MyParam1', 'MyParam2'};
 
% Write a new modified C3D and read back the data
ezc3dWrite('temporary.c3d', c3d);
c3d_to_compare = ezc3dRead('temporary.c3d');
 
% Print the header
fprintf('%% ---- HEADER ---- %%\n');
fprintf('Number of points = %d\n', c3d_to_compare.header.points.size);
fprintf('Point frame rate = %1.1f\n', c3d_to_compare.header.points.frameRate);
fprintf('Index of the first point frame = %d\n', c3d_to_compare.header.points.firstFrame);
fprintf('Index of the last point frame = %d\n', c3d_to_compare.header.points.lastFrame);
fprintf('\n');
fprintf('Number of analogs = %d\n', c3d_to_compare.header.analogs.size');
fprintf('Analog frame rate = %1.1f\n', c3d_to_compare.header.analogs.frameRate);
fprintf('Index of the first analog frame = %d\n', c3d_to_compare.header.analogs.firstFrame);
fprintf('Index of the last analog frame = %d\n', c3d_to_compare.header.analogs.lastFrame);
fprintf('\n');
fprintf('\n');

% Print the parameters
fprintf('%% ---- PARAMETERS ---- %%\n');
fprintf('Number of points = %d\n', c3d_to_compare.parameters.POINT.USED);
fprintf('Name of the points =\t');
    fprintf('%s\t', c3d_to_compare.parameters.POINT.LABELS{:}); 
    fprintf('\n');
fprintf('Point frame rate = %1.1f\n', c3d_to_compare.parameters.POINT.RATE);
fprintf('Number of frames = %d\n', c3d_to_compare.parameters.POINT.FRAMES);
fprintf('My point new Param =\t');
    fprintf('%d\t', c3d_to_compare.parameters.POINT.NEWPARAM);
    fprintf('\n');
fprintf('\n');
fprintf('Number of analogs = %d\n', c3d_to_compare.parameters.ANALOG.USED);
fprintf('Name of the analogs =\t');
    fprintf('%s\t', c3d_to_compare.parameters.ANALOG.LABELS{:}); 
    fprintf('\n');
fprintf('Analog frame rate = %1.1\n', c3d_to_compare.parameters.ANALOG.RATE);
fprintf('\n');
fprintf('My NewGroup new Param =\t')
    fprintf('%s\t', c3d_to_compare.parameters.NEWGROUP.NEWPARAM{:});
    fprintf('\n');
fprintf('\n');
fprintf('\n');

% Print the data
fprintf('%% ---- DATA ---- %%\n');
fprintf('See figures\n');
frameToPlot = 1;
figure('Name', '3d-Points');
plot3(c3d_to_compare.data.points(1,:,frameToPlot), ...
      c3d_to_compare.data.points(2,:,frameToPlot), ...
      c3d_to_compare.data.points(3,:,frameToPlot), 'k.'); 
axis equal
figure('Name', 'Analogs');
plot(c3d_to_compare.data.analogs);
