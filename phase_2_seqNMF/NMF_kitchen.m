function NMF_kitchen(L, K, lambdaL1H, lambda_bin, lambda)
    traj_len = 100;
    load('kitchen_data/kitchen_train.mat')
    X = double(s_a_pairs_normed);

    %% Fit with seqNMF
    [dim, len] = size(X);
    % lambdaH = 0.01; % Encourage event-based
    % lambdaL1H = 0.0;
    % lambdaW = 0.01;
    
    shg; clf
    disp('Running seqNMF on simulated data (2 simulated sequences + noise)')
    rng(2)
    [W,H] = seqNMF(X, traj_len, 'K',K, 'L', L,'lambda', lambda, 'lambdaL1H', ...
    lambdaL1H, "lambdabin", lambdabin, 'maxiter', 301, ...
                    'shift', 1, 'seed', seed,...
                    'W_init', 3, 'H_init', nan, 'showPlot', 1, 'savePlot', 1, 'start_indices', start_indices);

    %% Look at factors
    figure; SimpleWHPlot(W,H); title('SeqNMF reconstruction')
    saveas(gcf,sprintf('train_logs/seqNMF_reconstruction.png'));

    figure; SimpleWHPlot(W,H,X); title('SeqNMF factors, with raw data')
    saveas(gcf,sprintf('train_logs/seqNMF_factors_and_raw_data.png'));


end
