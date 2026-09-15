function [C,n]=Data()
load('H12'); X=H12(:,1:3); Y=H12(:,4:6);
n=size(X,1);
W=[100,100,100];
for i=1:n
    for j=1:n
        C(i,j)=norm(X(i,:)-(Y(j,:)+W))^(2);
    end
end
end


