function [C ,n] = Z_One_D()
% Reproducibility
rng(1);

% Total number of points
n = 3000;

% Number of clusters
num_clusters = 3;
points_per_cluster = n / num_clusters;

% Cluster centers (well-separated)
centers = [0, 0.5, 1];

% Standard deviation within each cluster
sigma = 0.3;

% Generate dataset X
X = [];
for i = 1:num_clusters
    Xi = centers(i) + sigma * randn(points_per_cluster, 1);
    X = [X; Xi];
end

% Sort (optional, useful for OT structure visualization)
X = sort(X);

% Generate second dataset Y (slightly shifted clusters)
shift = 3.5;
Y = [];
for i = 1:num_clusters
    Yi = centers(i) + shift + sigma * randn(points_per_cluster, 1);
    Y = [Y; Yi];
end
% Y =3*X;
% Y = sort(Y);

%% Non Uniform
X = (0.001:0.001:3).^2;
% X = (0.001:0.001:1).^2;
% X = (0.01:0.01:5).^2;
Y = X + 1;
n = size(X,2);


% % Uniform weights (can be modified if needed)
% a = ones(n,1) / n;
% b = ones(n,1) / n;

% Cost matrix (squared Euclidean distance)
C = (X - Y').^2;

% Visualization
% figure;
% subplot(2,1,1);
% plot(X, zeros(size(X)), 'bo');
% title('Dataset X (3 clusters)');
% ylim([-1,1]);
% 
% subplot(2,1,2);
% plot(Y, zeros(size(Y)), 'ro');
% title('Dataset Y (shifted clusters)');
% ylim([-1,1]);
% 
% % Optional: visualize cost matrix
% figure;
% imagesc(C);
% colorbar;
% title('Cost Matrix C = (X - Y)^2');
end


