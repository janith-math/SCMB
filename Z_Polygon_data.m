function [C,Exact]=Z_Polygon_data(n)
Y = zeros(2,n);
for i = 1:n
    y = [cos(2*pi*i/n),sin(2*pi*i/n)];
    Y(:,i)=y;
end
%%  Z is from using Translation
% Z = Y + 1;
%%  Z is from using Rotation
rng(1)
[Q_0,~]=qr(eye(2)+0.7*randn(2,2),0);
Q0=Q_0*diag([-1 1]');
Z=Q0*Y;                   % Rotate the point set Y
%%
C = zeros(n);
for i=1:n
    for j=1:n
        C(i,j)=norm(Y(:,i)-Z(:,j))^(2);
    end
end
Exact = sum(diag(C));
%%
% figure(1)
% Plot_Polygon(Y,n)
% % hold on
% figure(2)
% Plot_Polygon(Z,n)
end


function Plot_Polygon(Y,n)
% To close the shape, append the first point at the end
Y(:,n+1) = Y(:,1);  % Append the first x coordinate
% figure(1)
plot(Y(1,:),Y(2,:),'*b-',LineWidth=2)
title(['Plygon for n = ',num2str(n)])
grid on
end