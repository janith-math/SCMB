function [cost,x]=Z_linprog1(n,A)
f=A(:)';
b=1;

I=eye(n);
T=ones(1,n);
Upper=kron(I,T);
Bottom=kron(T,I);
Aeq=[Upper;Bottom];

beq=b*ones(2*n,1);
Aineq=[];
bineq=[];
lb=zeros(1,n^2);
ub=max(beq)*ones(1,n^2);
x = linprog(f,Aineq,bineq,Aeq,beq,lb,ub);
x=reshape(x,n,n);
cost=abs(sum(sum(A.*x)));
fprintf('LP optimal solution is = %3.5f\t \n',full(cost))
end

