function [X,t,ED] = SCMB_Diag(n, C, Exact, useGPU)
if nargin < 4, useGPU = false; end

dtype = "double";

% ---- GPU availability ----
if useGPU
    try
        gpuDevice; % #ok<NASGU>
    catch
        warning("GPU not available. Falling back to CPU.");
        useGPU = false;
    end
end

% ---- cast / move data ----
C = cast(C, dtype);
if useGPU, C = gpuArray(C); end

m = max(C(:));
t = 1 / m;
Iter = 50;

e  = ones(n,1, dtype);     if useGPU, e  = gpuArray(e); end
nu = ones(2*n,1, dtype);   if useGPU, nu = gpuArray(nu); end
ED = zeros(Iter,1, dtype); if useGPU, ED = gpuArray(ED); end

F = @(nu_local, K_local) sum(K_local * e) - sum(nu_local);

%% ---- Diagonal Preconditioner for the S-system ----
delta = cast(1e-2, dtype); if useGPU, delta = gpuArray(delta); end
d = [];  % diagonal preconditioner (GPU vector)
mu = 2;

for k = 1:Iter
    if k < 15%18-->t1.5 % 15--->t2, 8-->t4, 10--->t3, 8--t6
        flag = false;
    else
        if flag==1
           % use flag == 2 for preconditioner only once after k=--
           % use flag == 1 for preconditioner once for each t update
        else
        flag = true;
        end
    end
%===============================
    for kk = 1:10
        alpha = 1; if useGPU, alpha = gpuArray(alpha); end

        y = nu(1:n);
        z = nu(n+1:end);
        Q = y + z.';
        E = -t*C + Q;
        A = exp(E);
        cost   = F(nu, A);
        grad_f = -([A*e; A.'*e] - [e; e]);

        beta1 = grad_f(1:n);
        beta2 = grad_f(n+1:end);
        D1e = A*e;
        D2e = A.'*e;

        if flag==0
        %% ---- Matrix-free CG on S1 ----
            RHS = beta1 - A*(beta2./D2e);
            RHS = RHS - mean(RHS);
            [alpha1t,~] = Mf_SCG(RHS, A, D1e, D2e,delta);
            alpha2t = (beta2 - A.'*alpha1t) ./ D2e;
                 c = sum(alpha2t)/(2*n);
            alpha1 = alpha1t + c;
            alpha2 = alpha2t - c;
            nu_nt  = [alpha1; alpha2];
        else
 %% ---- PCG branch with diagonal preconditioner ----
        if flag == 1
           % if kk > 2
               % use previous d
               % fprintf('use previous d  \n');
           % else
               [d]=Diag_Prec(A,e,delta,n);
               % disp("===========================")
           % end
           flag = 1; % Use flag =1 if want to generate the preconditioner for each inner iteration.
        end
        %% Left-sided Preconditioner
            z1 = beta1 - A*(beta2 ./ D2e);
            z1 = z1 - mean(z1);
            [alpha1t, resvec3] = Mf_SPCG_Diag(z1, A, D1e, D2e, delta,d);
            alpha2t = (beta2 - A.'*alpha1t) ./ D2e;
                 c = sum(alpha2t)/(2*n);
            alpha1 = alpha1t + c;
            alpha2 = alpha2t - c;
            nu_nt  = [alpha1; alpha2];
%% Figure purpose
%             z1 = beta1 - A*(beta2 ./ D2e);
%             z1 = z1 - mean(z1);
%             [alpha12,resvec1] = Mf_SCG(z1, A, D1e, D2e,delta);
% Plot_Both_resvec(resvec1, resvec3)
        end
        % ---- Backtracking line search ----
        for kkk = 1:20
            nu1 = nu + alpha * nu_nt;
            y1 = nu1(1:n);
            z1 = nu1(n+1:end);
            A1 = exp(-t*C + (y1 + z1.'));
            cost1 = F(nu1, A1);
            if real(cost1) < real(cost)
                nu = nu1;
                y  = y1;
                z  = z1;
                break
            else
                alpha = alpha / 2;
            end
        end
        % ---- Update primal variable ----
        X = exp(-t*C + (y + z.'));
        % ---- Balancing stop ----
        Ebal = norm([sum(X,2)-e; sum(X,1).'-e], 2);
        if Ebal <= cast(1e-5, dtype) * sqrt(cast(n, dtype))
            break
        end
    end
    % ---- Primal-dual gap ----
    Primal_k = sum(C .* X, "all");
    Dual_k   = sum([y; z]) / t;
    ED(k) = abs(Primal_k - Dual_k) / cast(Exact, dtype);
    time(k)=toc;
    if ED(k) <= cast(1e-5, dtype)
        ED = ED(1:k);
        break
    end
    % ---- Barrier update ----
    t  = t * mu;
    nu = nu* mu;
    % fprintf(2,'t =============== %3.4f\t \n', t);
end
 Plot_Duality(time,ED,n,t,mu)
end


function [d]=Diag_Prec(A,e,delta,n)
 %% ---- PCG branch with diagonal preconditioner ----
    % ================= CPU-only sparse steps =================
    As = A_sparse_cpu(A,n);  
    % As = A;
    d1  = As * e;     % diagonal entries of D1
    d2  = As' * e;    % diagonal entries of D2
    d = (d1+delta) - (A.^2) * (1 ./ d2);
    % % Ldiag  = ((d1 + delta) - sum((1./(d2.')).*(As.^2), 2)); 
    % =========================================================
end


% =======================================================================
% CPU-only helper: build sparse top-k approximation of a dense matrix.
% =======================================================================
function Ks = A_sparse_cpu(A,n)
k = 20;    % 0,20,40,100
%   Keeps top-k entries per column by value, then symmetrizes and includes diag.
%   Output Ks is sparse and weighted by A: Ks(i,j)=A(i,j) on selected pattern.
[~, Sigma1] = maxk(A, k, 1);  % top-k per column
S = sparse(n,n);
for j = 1:n
    rows = Sigma1(:,j);
    S(rows, j) = 1;
end
S = S | speye(n) | S.';   % symmetrize + include diagonal
S = spones(S);
Ks = S .* sparse(A);
end


function Plot_Both_resvec(resvec1,resvec3)
figure
semilogy(resvec1,'b*-','LineWidth',2)
hold on
% semilogy(resvec2,'m*-','LineWidth',2)
semilogy(resvec3,'g*--','LineWidth',2)
xlabel('Iteration')
ylabel('Residual Norm ||r_k||')
title('Diagonal-Preconditioner Convergence')
legend('CG','PCG_{Diag}')
grid on
hold off
end

function Plot_Duality(time,E,n,~,mu)
figure
semilogy(time,E,'m*-','Linewidth',2)
title(['Duality gap dacay with time for n = ',num2str(n),' when t ==> ',num2str(mu)])
xlabel('CPU Time (s)')
ylabel('error')
legend('SCMB_{Diag}')
grid on
end