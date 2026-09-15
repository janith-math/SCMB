function [x, resvec] = Mf_SCG(b, A, D1e, D2e, delta)
% Matrix_free_SCG  GPU-capable Conjugate Gradient.

% Solves % S1 = diag(D1e) - A * diag(D2e)^(-1) * A' x = b

tol = 1e-6;
x = gpuArray.zeros(size(b), 'double');   %  initial guess

matvec = @(v) (D1e + delta) .* v - A * ((A' * v) ./ D2e);
%% Initial residual r = b - A*x
r = b - matvec(x);
% Enforce orthogonality to the nullspace (mean-zero)
r = r - mean(r);




p = r;
rsold = r' * r;
resvec = norm(r);   % store first residual
for i = 1:100
    Ap = matvec(p);
    denom = (p' * Ap);

    if abs(denom) <= eps(cast(1,"like",denom))
        break
    end

    alpha = rsold / denom;
    x = x + alpha * p;
    r = r - alpha * Ap;
    r = r - mean(r);




    rsnew = r' * r;
    resvec(i+1) = norm(r);   % store residual norm
    if sqrt(rsnew) < tol
        break
    end

    beta = (rsnew / rsold);
    p     = r + beta * p;
    rsold = rsnew;
end
end

