function bern_gradient_neg_loglik!(G, w, X, y, gamma)
    mu = X * w
    sigma_mu = 1 ./ (1 .+ exp.(-mu)) 
    G .= - X' * (gamma .* y) + X' * (gamma .* sigma_mu)
end

function bern_hessian_neg_loglik!(H, w, X, y, gamma)
    mu = X * w
    sigma_mu = 1 ./ (1 .+ exp.(-mu))  
    sigma_prime_mu = sigma_mu .* (1 .- sigma_mu)  
    D = Diagonal(gamma .* sigma_prime_mu)
    H .= X' * D * X
end

function bern_neg_loglik(w, X, y, gamma_vec)
    mu = X * w
    nll = - (gamma_vec .* y)' * mu + sum(gamma_vec .* softplus.(mu))
    return nll 
end

function bern_m_step_with_derivs(X, y, gamma, n_cols, n_states)
    W = fill(NaN, n_cols, n_states)

    for i in 1:n_states
        gamma_state = gamma[:, i]
        opts = optimize(
            w -> bern_neg_loglik(w, X, y, gamma_state),
            (G, w) -> bern_gradient_neg_loglik!(G, w, X, y, gamma_state),
            (H, w) -> bern_hessian_neg_loglik!(H, w, X, y, gamma_state),
            zeros(n_cols),
            NewtonTrustRegion(),
        )
        W[:, i] = opts.minimizer
    end

    return W
end

function gauss_m_step(X, Y, gamma, n_samp_per_trial)
    K, M = size(gamma, 2), size(Y, 2)
    W = zeros(size(X, 2), M, K)
    sig2 = zeros(M, K)
    big_gamma = vcat(_repeat_by_trial_length(n_samp_per_trial, gamma)...)
    big_gamma = big_gamma ./ sum(big_gamma, dims=2)

    for k in 1:K
        X_gamma = X .* big_gamma[:, k]
        XzX = X' * X_gamma
        XzY = X' * (Y .* big_gamma[:, k])
        for m in 1:M
            W[:, m, k] = (XzX ) \ XzY[:, m]
            residuals = Y[:, m] .- X * W[:, m, k]
            sig2[m, k] = sum(big_gamma[:, k] .* residuals .^ 2) / sum(big_gamma[:, k])
        end
    end

    return W, sig2
end

function prob_param_m_step!(prob_obj, e_quants)
    prob_obj.pi0 .= e_quants.pi0
    prob_obj.pi0 ./= sum(prob_obj.pi0)

    for i in 1:size(prob_obj.A, 1)
        N_i = e_quants.xi[i, :] + prob_obj.dir_prior[i, :]
        prob_obj.A[i, :] .= N_i ./ sum(N_i)
    end
end

function tdist_regression(X, Y, gamma, nu, rng_state, max_iter, tol)
    n_trials = size(X, 1)
    w_dim = size(X, 2)
    y_dim = size(Y, 2)
    W = rand(rng_state, Normal(0, 1), w_dim, y_dim) .* 0.1
    weights = zeros(n_trials, y_dim)
    sig2 = rand(rng_state, Exponential(0.5), y_dim)
    params_old = [vec(W); sig2]

    for iter in 1:max_iter
        for m in 1:y_dim
            residuals = Y[:, m] .- X * W[:, m]
            E_w = (nu[m] + 1) ./ (nu[m] .+ (residuals.^2 ./ sig2[m]))
            weights[:, m] = E_w .* gamma
            W[:, m] = (transpose(X) * (weights[:, m] .* X)) \ ( transpose(X) * (weights[:, m] .* Y[:, m]))
            sig2[m] = sum(weights[:, m] .* residuals.^2) / sum(weights[:, m])
        end
      
        params = [vec(W); sig2]
        param_diff = norm(params - params_old)
        if param_diff < tol
            println("TDist EM converged after $iter iterations.")
            break
        end
        params_old = params
    end

    return (
        W = W,
        sig2 = sig2,
        weights,
        nu = nu
        )
end

function tdist_m_step(X, Y, gamma, nu, n_samp_per_trial, rng_state, max_iter, tol)
    K, M = size(gamma, 2), size(Y, 2)
    W = zeros(size(X, 2), M, K)
    sig2 = zeros(M, K)
    big_gamma = vcat(_repeat_by_trial_length(n_samp_per_trial, gamma)...)
    big_gamma = big_gamma ./ sum(big_gamma, dims=2)

    for k in 1:K
        reg_result = tdist_regression(X, Y, big_gamma[:, k], nu, rng_state, max_iter, tol)
        W[:, :, k] = reg_result.W
        sig2[:, k] = reg_result.sig2
    end

    return (
        W = W,
        sig2 = sig2
        )
end
