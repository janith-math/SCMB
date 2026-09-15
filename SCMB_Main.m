clc; clear; close all;

% ===== GPU-enabled main script for SCMB =====
% Notes:
% 1) Use SINGLE on GPU for speed/memory unless you need double precision.
% 2) For large n, ensure your GPU has enough VRAM.
% 
n = 4000;
C = magic(n);            % Cost matrix 

% Exact = 3904688750;      % for magic(2500)
% Exact = 1389111750;     % for the magic(1950)
Exact=15995999487;     % for magic(4000)
% [C,n]=Data();     Exact=  7886799.26365;	      % Lung data

% [C,n] = Teapot_C();

% Exact = 173.5668;         % Teapot 100
% Exact = 374.665;          % Teapot 250
% Exact = 726.197;          % Teapot 500
% Exact = 1126.491;          % Teapot 800
% Exact = 1420.088;          % Teapot 1000
% Exact = 3489.091;          % Teapot 2500
% Exact = 6932.944;          % Teapot 5000
% Exact = 374.665;          % Teapot 10000
% Exact = 374.665;          % Teapot 12500
% Exact = 374.665;          % Teapot 20000
% Exact = 1; 

% [C ,n] = Z_One_D(); 
% Exact = n;
% [Exact,x_lpg]=Z_linprog1(n,C);

% n= 10;
% [C,Exact]=Z_Polygon_data(n);

useGPU = true;           % set false to run CPU version


% start = tic;
tic
[X, t, ED] = SCMB_Diag(n, C, Exact, useGPU);  
toc
% END_TIME = toc(start)

OPT_sol = sum(C .* X, "all");

fprintf(2,'The Exact Optimal solution is %3.4f\t \n', Exact);
fprintf(2,'The approximated Optimal solution is %3.4f\t \n', OPT_sol);
