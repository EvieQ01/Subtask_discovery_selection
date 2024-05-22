function W_init = initialize_W_driving(X,l, L, K, T, N)
    %%initialize_W
    % X: data matrix
    % L: number of lag dimension
    % K: number of clusters
    % T: number of total length
    % N: number of data dimension
    
    W_init = zeros(N, K, L);
    L_short = L;
    for k = 1:K
        is_picked = false;
        while is_picked == false
            rand_id_traj = randi(floor(T/l)-1, 1, 1) % 0 ~ T/traj_len-1. The pattern will start from No. x traj
            rand_id_subtraj = randi(ceil(l/L_short), 1, 1) % 1 ~ L. The pattern start from the y*L position of No. x traj
            temp = X(:, 1 + rand_id_traj * l + (rand_id_subtraj - 1) * L_short: ...
                        L + rand_id_traj * l + (rand_id_subtraj - 1) * L_short);
            
            % compare with other earlier clusters
            if k == 1
                is_picked = true;
                W_init(:, k, :) = temp;
            else
                is_picked = true;
                for kk = 1:k-1
                    norm(temp - squeeze(W_init(:, kk, :))) / sqrt(L)
                    if norm(temp - squeeze(W_init(:, kk, :))) / sqrt(L) < 0.3 %for non-kitchen env
                        is_picked = false;
                    end
                end
            end
            % if picked, add to W(k)
            if is_picked
                W_init(:, k, :) = temp;
            end
        end
    end
    % permute(W_init, [1 3 2])
    end