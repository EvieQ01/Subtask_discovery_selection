function C = reconstruct_option(W,H, plot_option, plot_length)
% ------------------------------------------------------------------------
% USAGE: C = helper.reconstruct_option(W,H, X_hat)
% ------------------------------------------------------------------------
% INPUTS
% W:      W is a NxKxL tensor which gives the neuron basis
%         functions which are used for the reconstructions. The L'th NxK slice
%         of W is the neural basis set for a lag of L.
%
% H:      H is a KxT matrix which gives timecourses for each factor 
% ------------------------------------------------------------------------
% OUTPUTS
% X_hat:  The reconstruction X_hat = W (*) H; 
% ------------------------------------------------------------------------
% Emily Mackevicius and Andrew Bahle

[N,K,L] = size(W);
[~,T] = size(H);
factor_value = rand(K,T) * 0.01;
C = zeros(K,T);


for t = 1:T
for tau = 1:L % go through every offset from 1:tau
    % tau_value = W(:, :, tau) * circshift(H,[0,tau-1]);
    % H_shifted = circshift(H,[0,tau-1]);

    for k = 1:K
        % sum(W(:, k, tau))
        factor_value(k, t) = factor_value(k, t) + sum(W(:, k, tau))* squeeze(H(k, max([t-tau+1, 1])));
    end
end
end
factor_value;
% make the sum of each factor = 1
C = factor_value ./ sum(factor_value, 1) + 1e-8;
% plot_length = 410;
if plot_option
    figure
    % Plot each row of C as a separate curve
    for k = 1:K
        plot(1:plot_length, squeeze(C(k, 1:1+plot_length-1)), 'LineWidth', 2)
        hold on
    end
    % Create labels for each curve
    labels = arrayfun(@(i) sprintf('Subtask %d', i), 1:K, 'UniformOutput', false);
    legend(labels, 'Location', 'best', 'FontSize', 25)
    hold off
    
    title('Subtask partition', 'FontSize', 30, 'FontWeight', 'bold');
    xlabel('Time step', 'FontSize', 24);
    % ylabel('Selection power', 'FontSize', 24);
    % legend({'sin(x)', 'cos(x)'}, 'Location', 'best');

    ax = gca;
    ax.FontSize = 25;
    ax.LineWidth = 1.5;
    grid on;
    xlim([1, plot_length]);
end

end