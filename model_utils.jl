add_dim(x) = reshape(x, (size(x)..., 1))

drop_dim(a) = dropdims(a, dims = (findall(size(a) .== 1)...,))

row_vector(x) = reshape(x, (1, length(x)))

function _repeat_by_trial_length(trial_lens, gamma)
    big_gammas = []
    n_state = size(gamma, 2)

    for (t, N_t) in enumerate(trial_lens)
        push!(big_gammas, ones(N_t, n_state) .* row_vector(gamma[t, :]))
    end

    return big_gammas
end

function one_hot(y)
    n = length(y)
    Y = zeros(n, 2)
    c = Int64.(y .+ 1)

    for i in 1:n
        Y[i, c[i]] = 1
    end

    return Y
end

function trial_inds(X)
    ii = vec(sum(X; dims=1) .!= 0)
    return (inds = ii, i = findall(ii))
end

function init_A_from_prior(rng_state, n_states, alpha_shared)
    _A = zeros(n_states, n_states)
    for i in 1:n_states
        _A[i, :] .= rand(rng_state, Dirichlet(alpha_shared[i, :]))
    end
    A = _A ./ sum(_A; dims=2)
    return A
end

function modified_tanh_alpha(alpha, x)
    if x == 1
        return x
    end
    return tanh(alpha * x) / tanh(alpha)
end

function generate_dir_prior(n_states, diag_term, off_diag_term)
    dir_prior = diagm(0 => diag_term * ones(n_states))
    for i in 1:n_states
        for j in 1:n_states
            if i != j
                dir_prior[i, j] = off_diag_term
            end
        end
    end
    return dir_prior
end

function init_gauss(rng_state, x, y, n_states)
    m = size(y, 2)
    W = hcat([coef(lm(x, y[:, i])) for i in 1:m]...)
    ijk = (size(W, 1), size(W, 2), n_states)
    W_states = W .+ rand(Normal(0, 1), ijk)
    sig2 = [var(y[:, i]) for i in 1:m]
    sig2_states = sig2 .+ rand(rng_state, Exponential(0.5), (m, n_states))
    return (W = W_states, sig2 = sig2_states)
end

function init_bern(rng_state, x, y, n_states)
    n_cols_bern = size(x, 2)
    w_bern = coef(glm(x, y, Binomial(), LogitLink()))
    W_bern = add_dim(w_bern) .+ rand(rng_state, Normal(0, 1), n_cols_bern, n_states)
    return W_bern
end
