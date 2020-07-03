clc
clear
close all
% addpath to ezc3d here

% Load a c3d that has force plaform
[c3d, all_pf] = ezc3dRead('../c3dFiles/ezc3d-testFiles-master/ezc3d-testFiles-master/Qualisys.c3d');

fprintf('Number of force platform = %d\n', size(all_pf, 1));
fprintf('\n');
fprintf('Printing information and data for force platform 1\n');
pf1 = all_pf(1);
fprintf('\n');

% Units
fprintf('Force unit = %s\n', pf1.unit_force');
fprintf('Moment unit = %s\n', pf1.unit_moment');
fprintf('Center of pressure unit = %s\n', pf1.unit_position');
fprintf('\n');

% Position of pf
fprintf('Position of origin = [%1.3f, %1.3f, %1.3f]\n', pf1.origin);
fprintf('Position of corners = [%1.3f, %1.3f, %1.3f]\n', pf1.corners);
fprintf('\n');

%Calibration matrix
fprintf('Calibation matrix = \n');
fprintf('%1.3f, %1.3f, %1.3f, %1.3f, %1.3f, %1.3f\n', pf1.cal_matrix);
fprintf('\n');

% Data at 3 different time
frames = [1, 11, 1001, size(pf1.force, 2)];
fprintf('Data (in global reference frame) at frames = [%d, %d, %d, %d]\n', frames);
fprintf('Force = [%1.3f, %1.3f, %1.3f]\n', pf1.force(:, frames));
fprintf('Moment = [%1.3f, %1.3f, %1.3f]\n', pf1.moment(:, frames));
fprintf('Center of pressure = [%1.3f, %1.3f, %1.3f]\n', pf1.center_of_pressure(:, frames));
fprintf('Moment at CoP = [%1.3f, %1.3f, %1.3f]\n', pf1.Tz(:, frames));
