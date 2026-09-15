function [X, f1, J, I_1] = ET_1_GPU(X, n, useGPU)
%ET_1  GPU-friendly early termination.
%
% Original code produced a sparse permutation matrix (CPU). On GPU, we return
% a FULL permutation matrix (still nonnegative and doubly stochastic).
%
% This version also fixes the non-scalar IF condition in the original ET_1.

if nargin < 3, useGPU = isa(X, "gpuArray"); end

v1 = 0.5;

[L, J] = max(X, [], 2);

% Ensure J is on CPU for uniqueness check (unique on GPU may be slower/limited)
J_cpu = gather(J);

if numel(J_cpu) == numel(unique(J_cpu))

    % Build Y by removing max element from each column of X'
    Y_1 = X.';                      % n-by-n

    % Linear index of maxima along each column (dimension 1 of Y_1)
    [~, I_1] = max(Y_1, [], 1, 'linear'); % 1-by-n linear indices

    % Remove these elements and reshape
    Y_1(I_1) = [];
    Y_2 = reshape(Y_1, n-1, n);
    Y   = Y_2.';                    % n-by-(n-1)

    % Compare rowwise: Y(i, :) <= v1 * L(i)
    cond = all(Y <= v1 * L, "all");

    if gather(cond)
        % Build permutation matrix corresponding to assignment J
        if useGPU
            X0 = gpuArray.zeros(n, n, "like", X);
        else
            X0 = zeros(n, n, "like", X);
        end
        idx = sub2ind([n n], (1:n).', J_cpu);
        X0(idx) = 1;

        X  = X0;     % full matrix (GPU-safe)
        f1 = 1;
    else
        f1 = 0;
    end
else
    f1 = 0;
    I_1 = 0;
end
end
