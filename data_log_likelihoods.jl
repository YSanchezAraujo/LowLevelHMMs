function compute_gauss_loglikelihoods!(loglik_obs_gauss, X, Y, W, sig2, n_steps, n_states)
    D = size(Y[1], 2)
    for t in 1:n_steps
        log_probs = zeros(n_states, D)
        N_t = size(Y[t], 1)
        for s in 1:n_states
            residuals = Y[t] .- X[t] * W[:, :, s]
            for m in 1:D
                distri = Normal(0., sqrt(sig2[m, s]))
                log_probs[s, m] = sum(logpdf.(distri, residuals[:, m]))
            end
        end
        total_log_prob = drop_dim(sum(log_probs, dims=2))
        loglik_obs_gauss[t, :] .= total_log_prob .- maximum(total_log_prob)
    end
end

function compute_bern_loglikelihoods!(loglik_obs_bern, X, y, W, n_steps, n_class, n_states)
    Y = one_hot(y)'
    XW = zeros(n_steps, n_class, n_states);
    XW[:, 2, :] = X * W;
    logpy_bern = XW .- logsumexp(XW; dims=2);
    loglik_obs_bern .= drop_dim(sum(logpy_bern .* Y'; dims=2))
end
