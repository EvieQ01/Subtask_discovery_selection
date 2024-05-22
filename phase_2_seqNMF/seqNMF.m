function [W, H, cost,loadings,power] = seqNMF(X, trajLen, varargin)
%
% USAGE: 
%
% [W, H, cost, loadings, power] = seqNMF(X, ...    % X is the data matrix
%       'K', 10, 'L', 20, 'lambda', .1, ...        % Other inputs optional
%       'W_init', W_init, 'H_init', H_init, ...
%       'showPlot', 1, 'maxiter', 20, 'tolerance', -Inf, 'shift', 1, ... 
%        'lambdaL1H', 0, ...
%       'lambdaOrthoH', 0 , 'M', M)
%
% ------------------------------------------------------------------------
% DESCRIPTION:
%
%   Factorizes the NxT data matrix X into K factors 
%   Factor exemplars are returned in the NxKxL tensor W
%   Factor timecourses are returned in the KxT matrix H
%
%                                    ----------    
%                                L  /         /|
%                                  /         / |
%        ----------------         /---------/  |          ----------------
%        |              |         |         |  |          |              |
%      N |      X       |   =   N |    W    |  /   (*)  K |      H       |           
%        |              |         |         | /           |              |
%        ----------------         /----------/            ----------------
%               T                      K                         T
% See paper: 
%   XXXXXXXXXXXXXXXXX
%
% ------------------------------------------------------------------------
%
% INPUTS:
%
% Name              Default                             Description
%  X                                                    Data matrix (NxT) to factorize
% 'K'               10                                  Number of factors
% 'L'               100                                 Length (timebins) of each factor exemplar
% 'lambda'          .001                                Regularization parameter
% 'W_init'          max(X(:))*rand(N,K,L)               Initial W
% 'H_init'          max(X(:))*rand(K,T)./(sqrt(T/3))    Initial H (rows have norm ~1 if max(data) is 1)
% 'showPlot'        1                                   Plot every iteration? no=0
% 'maxiter'         100                                 Maximum # iterations to run
% 'tolerance'       -Inf                                Stop if improved less than this;  Set to -Inf to always run maxiter
% 'shift'           1                                   Shift factors to center; Helps avoid local minima
% 'lambdaL1H'       0                                   L1 sparsity parameter; Increase to make H's more sparse
% 'W_fixed'         0                                   Fix W during the fitting proceedure   
% 'SortFactors'     1                                   Sort factors by loadings
% 'lambdaOrthoH'    0                                   ||HSH^T||_1,i~=j; Encourages events-based factorizations
% 'lambdabin'       0                                   ||H^T(1-H)||_2^2; binary constraint
% 'useWupdate'      1                                   Wupdate for cross orthogonality often doesn't change results much, and can be slow, so option to remove  

% ------------------------------------------------------------------------
% OUTPUTS:
%
% W                         NxKxL tensor containing factor exemplars
% H                         KxT matrix containing factor timecourses
% cost                      1x(#Iterations+1) vector containing 
%                               reconstruction error at each iteration. 
%                               cost(1) is error before 1st iteration.
% loadings                  1xK vector containing loading of each factor 
%                               (Fraction power in data explained by each factor)
% power                     Fraction power in data explained 
%                               by whole reconstruction
%
%                           Note, if doing fit with masked (held-out) data,
%                               the cost and power do not include masked
%                               (M==0) test set elements
% ------------------------------------------------------------------------
% CREDITS:
%   Emily Mackevicius and Andrew Bahle, 2/1/2018
%
%   Original CNMF algorithm: Paris Smaragdis 2004
%   (https://link.springer.com/chapter/10.1007/978-3-540-30110-3_63)
%   Adapted from NMF toolbox by Colin Vaz 2015 (http://sail.usc.edu)
%
%   Please cite our paper: 
%       https://www.biorxiv.org/content/early/2018/03/02/273128
%% parse function inputs

% Check that we have non-negative data
if min(X(:)) < 0
    error('Negative values in data!');
end

% Parse inputs
[X,N,T,K,L,params] = parse_seqNMF_params(X, varargin);

%% initialize
W = params.W_init;
H = params.H_init;

Xhat = helper.reconstruct(W, H); 
mask = find(params.M == 0); % find masked (held-out) indices 
X(mask) = Xhat(mask); % replace data at masked elements with reconstruction, so masked datapoints do not effect fit

smoothkernel = ones(1,(2*L)-1);  % for factor competition
smallnum = max(X(:))*1e-6; 
lasttime = 0;

% Calculate initial cost
cost = zeros(params.maxiter+1, 1);
cost(1) = sqrt(mean((X(:)-Xhat(:)).^2));

start_bin_loss_iter = 30;
for iter = 1 : params.maxiter
    % Stopping criteria... Stop if reach maxiter or if change in cost function is less than the tolerance
    if (iter == params.maxiter) || ((iter>5) && (cost(iter+1)+params.tolerance)>mean(cost((iter-5):iter)))
        cost = cost(1 : iter+1);  % trim vector
        lasttime = 1; 
        if iter>1
            params.lambda = 0; % Do one final CNMF iteration (no regularization, just prioritize reconstruction)
        end
    end
    
    % Compute terms for standard CNMF H update 
    WTX = zeros(K, T);
    WTXhat = zeros(K, T);
    for l = 1 : L
        X_shifted = circshift(X,[0,-l+1]); 
        Xhat_shifted = circshift(Xhat,[0,-l+1]); 
        WTX = WTX + W(:, :, l)' * X_shifted;
        WTXhat = WTXhat + W(:, :, l)' * Xhat_shifted;
    end   
         
    % Compute regularization terms for H update
    if params.lambda>0
        dRdH = params.lambda.*(~eye(K))*conv2(WTX, smoothkernel, 'same');  
    else 
        dRdH = 0; 
    end
    if params.lambdaOrthoH>0
        dHHdH = params.lambdaOrthoH*(~eye(K))*conv2(H, smoothkernel, 'same');
    else
        dHHdH = 0;
    end
    
    % add bin loss
    if params.lambdabin > 0 && iter > start_bin_loss_iter
        T_0 = ones(size(H)) - H;
        % T_0 * T_0' * H
        % H * H' * T_0
        dBHdH = params.lambdabin*(T_0 .* T_0 .* H - H .* H .* T_0);

        % Clip the values of dBHdH to the range [0, 1]
        dBHdH = min(1, dBHdH);
    else
        dBHdH = 0;
    end

    %%%% All derivatives on H
    
    % iter
    % dHHdH
    % dBHdH
    dRdH = dRdH + params.lambdaL1H + dHHdH +  dBHdH; % include L1 sparsity, if specified
    
    
    % Update H
    H = H .* WTX ./ (WTXhat + dRdH +eps);

    % Shift to center factors
    if params.shift
        [W, H] = helper.shiftFactors(W, H);  
        W = W+smallnum; % add small number to shifted W's, since multiplicative update cannot effect 0's
    end
    
    % Renormalize so rows of H have constant energy
    norms = sqrt(sum(H.^2, 2))';

    % change norm to sqrt(T/L)
    H = diag(sqrt(T/L) ./ (norms+eps)) * H;
    for l = 1 : L
        W(:, :, l) = W(:, :, l) * diag(norms) / sqrt(T/L);
    end 

    if iter > start_bin_loss_iter
        % scaling so max loading is 1
        maxs = max(H, [], 2)';
        H = diag(1 ./ (maxs+eps)) * H;
        for l = 1 : L
            W(:, :, l) = W(:, :, l) * diag(maxs);
        end 
        
        if not(isnan(params.start_indices))
            % scaling so max loading at start index of a trajectory is 1
            H_starts = H(:, params.start_indices + L); % (K, 20)
            % max(H_starts, [], 1)
            % H(:, params.start_indices + L) = (0.4 + 0.002 * iter )./ (max(H_starts, [], 1)+eps) .* H_starts;
            H(:, params.start_indices + L) = H_starts./ (sum(H_starts, 1)+eps);
        end
    end

    if ~params.W_fixed
    % Update each Wl separately
        Xhat = helper.reconstruct(W, H); 
        mask = find(params.M == 0); % find masked (held-out) indices 
        X(mask) = Xhat(mask); % replace data at masked elements with reconstruction, so masked datapoints do not effect fit
        if params.lambda>0 && params.useWupdate
            XS = conv2(X, smoothkernel, 'same'); 
        end
        for l = 1 : L % could parallelize to speed up for long L
            % Compute terms for standard CNMF W update
            H_shifted = circshift(H,[0,l-1]);
            XHT = X * H_shifted';
            XhatHT = Xhat * H_shifted';

            % Compute regularization terms for W update
            if params.lambda>0 && params.useWupdate % Often get similar results with just H update, so option to skip W update
                dRdW = params.lambda.*XS*(H_shifted')*(~eye(K)); 
            else
                dRdW = 0;
            end
            % Update W
            mem = W(:, :, l); % smooth out W update
            W(:, :, l) = W(:, :, l) .* XHT ./ (XhatHT + dRdW + eps);
            W(:, :, l) = (W(:, :, l) + mem) ./ 2;
        end
    end
    % Calculate cost for this iteration
    Xhat = helper.reconstruct(W, H);    
    mask = find(params.M == 0); % find masked (held-out) indices 
    X(mask) = Xhat(mask); % replace data at masked elements with reconstruction, so masked datapoints do not effect fit
    cost(iter+1) = sqrt(mean((X(:)-Xhat(:)).^2));

    % Plot to show progress
    if params.showPlot 
        SimpleWHPlot(W, H, Xhat,0); 
        title(sprintf('iteration #%i',iter));
        % if params.savePlot
        %     saveas(gcf,sprintf('train_logs/iters/seqNMF_iter%i.png',iter));
        % end
        drawnow
    end
    if params.savePlot 
        if (iter == params.maxiter)
        SimpleWHPlot(W, H, Xhat,0); 
        % title(sprintf('OrthoH: %.3f, H1: %.3f, Hbin: %.3f', params.lambdaOrthoH, params.lambdaL1H, params.lambdabin));
        folderPath = sprintf('train_logs/seed%d/data_length_%d', params.seed, T - 2 * L);
        if ~exist(folderPath, 'dir')
        mkdir(folderPath);
        end
        saveas(gcf,sprintf('train_logs/seed%d/data_length_%d/OrthoH_%.3f_H1_%.3f_Hbin_%.3f.png', params.seed, T - 2 * L, params.lambdaOrthoH, params.lambdaL1H, params.lambdabin)); 
        end
    end
    
    if lasttime
        break
    end
end
   
% Undo zeropadding by truncating X, Xhat and H
X = X(:,L+1:end-L);
Xhat = Xhat(:,L+1:end-L);
H = H(:,L+1:end-L);

% Compute explained power of whole reconstruction and each factor
power = (sum(X(:).^2)-sum((X(:)-Xhat(:)).^2))/sum(X(:).^2);  % fraction power explained by whole reconstruction
[loadings,ind] = sort(helper.computeLoadingPercentPower(X,W,H),'descend'); % fraction power explained by each factor

% sort factors by loading power
if params.SortFactors
    W = W(:,ind,:);
    H = H(ind,:);
end

    function [X,N,T,K,L,params] = parse_seqNMF_params(X, inputs);
        % parse inputs, set unspecified parameters to the defaults
        
        % Get data dimensions
        [N, T] = size(X);

        p = inputParser; % 
        %USAGE: addOptional(p,'parametername',defaultvalue);
        addOptional(p,'K',10);
        addOptional(p,'L',100);
        addOptional(p,'lambda',.001);
        addOptional(p,'showPlot',1);
        addOptional(p,'maxiter',100);
        addOptional(p,'tolerance',-Inf);
        addOptional(p,'shift',1);
        addOptional(p,'lambdaL1H',0);
        addOptional(p,'W_fixed',0);
        addOptional(p,'W_init', nan); % depends on K--initialize post parse
        addOptional(p,'H_init', nan); % depends on K--initialize post parse
        addOptional(p,'SortFactors', 1); % sort factors by loading?
        addOptional(p,'lambdaOrthoH',0); % for this regularization: ||HSH^T||_1,i~=j

        addOptional(p,'lambdabin',0); % for this regularization: ||H(1-H)||_2^2
        addOptional(p,'useWupdate',1); % W update for cross orthogonality often doesn't change results much, and can be slow, so option to skip it 
        addOptional(p,'M',nan); % Masking matrix: default is ones; set elements to zero to hold out as masked test set
        addOptional(p,'savePlot',1); % save plots to train_logs folder
        addOptional(p,'start_indices',nan); % the starting index of the trajectory
        addOptional(p,'seed', 0); % the seed index of the trajectory
        addOptional(p,'reward', nan); % the true reward of the trajectory
        parse(p,inputs{:});
        L = p.Results.L; 
        K = p.Results.K; 
        params = p.Results; 
        
        % zeropad data by L
        X = [zeros(N,L),X,zeros(N,L)];
        [N, T] = size(X);

        % initialize W_init and H_init, if not provided
        indices = params.start_indices;
        if params.W_init == 0 % random initialization
            params.W_init = max(X(:))*rand(N, K, L);
            % params.W_init = helper.initialize_W(X, trajLen, L, K, T, N, indices);
        elseif params.W_init == 2 % driving initialization
            %params.W_init = max(X(:))*rand(N, K, L);
            params.W_init = helper.initialize_W_driving(X, trajLen, L, K, T, N);
        elseif params.W_init == 3 % Kitchen initialization
            %params.W_init = max(X(:))*rand(N, K, L);
            params.W_init = helper.initialize_W(X, trajLen, L, K, T, N, indices);
        end
        if isnan(params.H_init) || params.H_init == 0
            params.H_init = max(X(:))*rand(K,T)./(sqrt(T/3)); % normalize so frobenius norm of each row ~ 1
            if not(isnan(params.start_indices))
                params.H_init(:, indices + L) = ones(K, length(indices)) / K;
            end
        else size(params.H_init) == [K, T - 2 * L]
            params.H_init = [zeros(K,L),params.H_init,zeros(K,L)];
        if isnan(params.M)
            params.M = ones(N,T);
        else
            params.M = [ones(N,L),params.M,ones(N,L)];
        end
    end
end