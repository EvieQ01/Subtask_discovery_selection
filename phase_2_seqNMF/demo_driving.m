load('driving.mat')
% Seq_NMF on driving
traj_len = 120
state_dim = 4;
X = double([state_normed(1:4,:); action_normed]);
action_normed = abs(action_normed);
% X = double([state_normed(1:2,1:240); action_normed(:,1:240)]);

%% Fit with seqNMF
K = 5;
L = 40;
[dim, len] = size(X)
lambda =.0001;
lambdaH = 0.00; % Encourage event-based
lambdaL1H = 0.01;
lambdabin = 0.01;
shg; clf
for seed = 1:50
seed
rng(seed)
[W,H] = seqNMF(X, traj_len, 'K',K, 'L', L,'lambda', lambda,'lambdaOrthoH', lambdaH, 'lambdaL1H', ...
                lambdaL1H, "lambdabin", lambdabin, 'maxiter', 301, ...
                                'shift', 1, 'seed', seed,...
                                'W_init', 2, 'H_init', nan, 'showPlot', 1, 'savePlot', 1, 'start_indices', start_indices);

H_bin  = H;
thresh = 0.5;
H_bin(H_bin > thresh) = 1;
H_bin(H_bin < thresh) = 0;
plot_option = 1;
C = helper.reconstruct_option(W,H_bin, plot_option, 210);

saveas(gcf,sprintf('train_logs/seed%d/data_length_%d/OrthoH_%.3f_H1_%.3f_Hbin_%.3f_option.png', seed, length(X),  lambdaH, lambdaL1H, lambdabin)); 

str = sprintf('train_logs/seed%d/data_length_%d/OrthoH_%.3f_H1_%.3f_Hbin_%.3f_C.mat', seed, length(X), lambda, lambdaL1H, lambdabin);
save(str, 'C')
str = sprintf('train_logs/seed%d/data_length_%d/OrthoH_%.3f_H1_%.3f_Hbin_%.3f_W.mat', seed, length(X), lambda, lambdaL1H, lambdabin);
save(str, 'W')
str = sprintf('train_logs/seed%d/data_length_%d/OrthoH_%.3f_H1_%.3f_Hbin_%.3f_H.mat', seed, length(X), lambda, lambdaL1H, lambdabin);
save(str, 'H')
end