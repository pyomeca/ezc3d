clc
clear
close all
fprintf('Matlab tests started..\n\n')

fprintf('Load a c3d with force platform and compare to known values..\n\n')
[c3d, all_pf] = ezc3dRead('../c3dFiles/ezc3d-testFiles-master/ezc3d-testFiles-master/Qualisys.c3d');

assert(size(all_pf, 1) == 2);
assert(size(all_pf, 2) == 1);

% Frames
assert(size(all_pf(1).force, 2) == 3400)
assert(size(all_pf(1).moment, 2) == 3400)
assert(size(all_pf(1).center_of_pressure, 2) == 3400)
assert(size(all_pf(1).Tz, 2) == 3400)

assert(size(all_pf(2).force, 2) == 3400)
assert(size(all_pf(2).moment, 2) == 3400)
assert(size(all_pf(2).center_of_pressure, 2) == 3400)
assert(size(all_pf(2).Tz, 2) == 3400)

% Units
assert(strcmp(all_pf(1).unit_force, 'N'))
assert(strcmp(all_pf(1).unit_moment, 'Nmm'))
assert(strcmp(all_pf(1).unit_position, 'mm'))

assert(strcmp(all_pf(2).unit_force, 'N'))
assert(strcmp(all_pf(2).unit_moment, 'Nmm'))
assert(strcmp(all_pf(2).unit_position, 'mm'))

% Position of pf
assert(sum(abs(all_pf(1).origin - [1.524,  -0.762, -34.036]')) < 1e-3)
assert(sum(sum(abs(all_pf(1).corners - [508, 508, 0, 0; 464, 0, 0, 464; 0, 0, 0, 0]))) < 1e-3)

assert(sum(abs(all_pf(2).origin - [1.016,  0, -36.322]')) < 1e-3)
assert(sum(sum(abs(all_pf(2).corners - [1017, 1017, 509, 509; 464, 0, 0, 464; 0, 0, 0, 0]))) < 1e-3)

% Calibration matrix
assert(sum(sum(abs(all_pf(1).cal_matrix - zeros(6, 6)))) < 1e-3)
assert(sum(sum(abs(all_pf(2).cal_matrix - zeros(6, 6)))) < 1e-3)

% Data at 3 different time
frames = [1, 1001, 3400];
expected_force = [0.140,  106.480, -0.140; 0.046, -66.407, -0.138; -0.184, 763.647, 0.367];
expected_moment = [20.868, 54768.655, 51.780; -4.623, -24103.676, 4.483; -29.393, -12229.124, -29.960];
expected_cop = [228.813, 285.564, 241.787; 118.296, 303.720, 373.071; 0, 0, 0];
expected_Tz = [0, 0, 0; 0, 0, 0; -44.141, -2496.299, -51.390];
assert(sum(sum(abs(all_pf(1).force(:, frames) - expected_force))) < 1e3)
assert(sum(sum(abs(all_pf(1).moment(:, frames) - expected_moment))) < 1e3)
assert(sum(sum(abs(all_pf(1).center_of_pressure(:, frames) - expected_cop))) < 1e3)
assert(sum(sum(abs(all_pf(1).Tz(:, frames) - expected_Tz))) < 1e3)

expected_force = [0.046, 0.232,  0.185; -0.185, -0.184, -0.046; 0.723,  0.361, 0.542];
expected_moment = [49.366, 68.671, 16.708; -96.907, -46.501, 50.403; 0.047, -19.720, 30.122];
expected_cop = [897.0422, 891.673, 670.044; 300.283, 422.019, 262.813; 0, 0, 0];
expected_Tz = [0, 0, 0; 0, 0, 0; 27.944, 48.016, 31.545];
assert(sum(sum(abs(all_pf(2).force(:, frames) - expected_force))) < 1e3)
assert(sum(sum(abs(all_pf(2).moment(:, frames) - expected_moment))) < 1e3)
assert(sum(sum(abs(all_pf(2).center_of_pressure(:, frames) - expected_cop))) < 1e3)
assert(sum(sum(abs(all_pf(2).Tz(:, frames) - expected_Tz))) < 1e3)

fprintf('Done\n');

fprintf('\nMatlab tests successfully completed!\n')
