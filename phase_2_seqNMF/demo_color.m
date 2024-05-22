load('color_10.mat')
% Seq_NMF on color 
one_hot_encode = false;
use_action = true;

if one_hot_encode
    num_labels = 3;
    one_hot_encoded = zeros(num_labels, T); % Initialize the one-hot encoded matrix
    
    for i = 1:T
        one_hot_encoded(state(i), i) = 1; % Set the corresponding element to 1
    end

    state_onehot = one_hot_encoded;
    X = double(state_onehot);
elseif use_action
    X = double([state_onehot; action_onehot]);
else
    X = double(state_onehot);
end

state_dim = 3;
seed=1;
%% Fit with seqNMF
K = 2;
L = 10;
T = length(state);
lambda =.001;
lambdaH = 0.00; % Encourage event-based
lambdaL1H = 0.01;
lambdaW = 0.0;
lambdabin = 0.01;
shg; clf
for seed=2
rng(seed)
[W,H] = seqNMF(X,'K',K, 'L', L,'lambda', lambda,'lambdaOrthoH', lambdaH, 'lambdaL1H', ...
                lambdaL1H, 'lambdaOrthoW', lambdaW, "lambdabin", lambdabin, 'maxiter', 301, ...
                'W_init', 0, 'H_init', 0);
%[W,H] = seqNMF(X,'K',K, 'L', L,'lambda', lambda,'lambdaOrthoH', lambdaH)
%% Look at factors
figure; SimpleWHPlot(W,H); title('SeqNMF reconstruction')
figure; SimpleWHPlot(W,H,X); title('SeqNMF factors, with raw data')

H_bin  = H;
thresh = 0.5;
H_bin(H_bin > thresh) = 1;
H_bin(H_bin < thresh) = 0;
plot_option = 1;
C = helper.reconstruct_option(W,H_bin, plot_option, 101)
save(strcat('C_est_color_', num2str(L), '_seed', num2str(seed), '.mat'),'C');
end