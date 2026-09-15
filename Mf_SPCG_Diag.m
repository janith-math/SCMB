function [x,resvec] = Mf_SPCG_Diag(b, A, D1e, D2e, delta, d)
% Matrix_free_SPCG  Preconditioned CG for Schur complement action.
%
% Solves (approximately) S x = b with
%   S(v) = D1e .* v - A * ((A' * v) ./ D2e)

tol = 1e-6;
x = gpuArray.zeros(size(b), 'double');   %  initial guess

matvec = @(v) (D1e+delta) .* v - A * ((A' * v) ./ D2e);

%% Initial residual r = b - A*x
r = b - matvec(x);
% Enforce orthogonality to the nullspace (mean-zero)
r = r - mean(r);

Minv  = 1 ./ gather(d);
z1 = r .* Minv;
z  = z1 - mean(z1);

p   = z;
zr0 = z' * r;
resvec = norm(r);   % store first residual
for i = 1:100
     Ap = matvec(p);
    denom = (p' * Ap);

    if abs(denom) <= eps(cast(1,"like",denom))
        break
    end

    alpha = zr0 / denom;
    x = x + alpha * p;
    r  = r - alpha * Ap;
    r = r - mean(r);

    z1 = r .* Minv;
    z  = z1 - mean(z1);

    zr = z' * r;
    resvec(i+1) = norm(r);   % store residual norm
    if sqrt(abs(zr)) < tol
        break
    end
    beta = zr / zr0;
    p = z + beta * p;
    zr0  = zr;
end
end
