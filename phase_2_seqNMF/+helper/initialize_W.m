function W_init = initialize_W(X, l, L, K, T, N, start_indices)
%%initialize_W
% X: data matrix
% l: length of each traj
% L: number of lag dimension
% K: number of clusters
% T: number of total length
% N: number of data dimension

% W_init = zeros(N, K, L);
W_init = rand(N, K, L) * 0.1;
L_short = floor(L / 2);
for k = 1:K
    is_picked = false;
    while is_picked == false
        rand_id_traj = randi(length(start_indices) -1 , 1, 1); % 0 ~ T/traj_len-1. The pattern will start from No. x traj
        rand_id_subtraj = randi(floor(l/L_short), 1, 1); % 1 ~ L. The pattern start from the y*L position of No. x traj
        temp = X(:, start_indices(rand_id_traj) + (rand_id_subtraj - 1) * L_short: ...
        start_indices(rand_id_traj + 1) - 1 + (rand_id_subtraj - 1) * L_short);
            % start_indices(rand_id_traj) + (rand_id_subtraj ) * L_short);
        
        [num_rows, num_cols] = size(temp);
        if num_cols < L
            % padd with zeros
            temp = cat(2, temp, zeros(N, L - num_cols));
        end  
        if num_cols > L
            % truncate
            temp = temp(:, 1:L);
        end
        
        % compare with other earlier clusters
        is_picked = true;
        if k > 1
            for kk = 1:k-1
                norm(temp(:, 1:L) - squeeze(W_init(:, kk, 1:L))) / sqrt(L);
                if norm(temp(:, 1:L) - squeeze(W_init(:, kk, 1:L))) / sqrt(L) < .6 %0.2 for non-kitchen env
                    is_picked = false;
                end
            end
        end
        % if picked, add to W(k)
        if is_picked
            W_init(:, k, :) = temp;
            % num_cols
            rand_id_traj
            rand_id_subtraj
        end
    end
end
permute(W_init, [1 3 2])
end