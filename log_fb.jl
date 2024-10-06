struct log_forward
    log_alpha::Matrix
    log_c::Vector
end

struct log_backward
    log_beta::Matrix
end

struct prob_param
    A::Matrix
    pi0::Vector
    dir_prior::Matrix 
end

function log_normalize(log_prob::Vector)
    log_c = logsumexp(log_prob)
    return log_prob .- log_c, log_c
end

function log_forward_message!(log_f_obj::log_forward, log_lik_obs::Matrix, log_pi0::Vector, log_A::Matrix)
    n_steps = size(log_lik_obs, 1)
    k = size(log_lik_obs, 2)
    log_alpha_t, log_c_t = log_normalize(log_lik_obs[1, :] .+ log_pi0)
    log_f_obj.log_alpha[1, :] .= log_alpha_t
    log_f_obj.log_c[1] = log_c_t

    for t in 2:n_steps
        log_alpha_t, log_c_t = log_normalize(log_lik_obs[t, :] .+ vec(logsumexp(log_A .+ log_f_obj.log_alpha[t-1, :]; dims=1)))
        log_f_obj.log_alpha[t, :] .= log_alpha_t
        log_f_obj.log_c[t] = log_c_t
    end

end

function log_backward_message!(log_b_obj::log_backward, log_lik_obs::Matrix, log_A::Matrix, log_c::Vector)
    log_b_obj.log_beta[end, :] .= 0.0

    for t in size(log_lik_obs, 1)-1:-1:1
        log_b_obj.log_beta[t, :] .= (
            vec(logsumexp(transpose(log_A) .+ (log_b_obj.log_beta[t+1, :] .+ log_lik_obs[t+1, :]); dims=1)) .- log_c[t+1]
        )
    end
end

function expectations(log_f_obj::log_forward, log_b_obj::log_backward, log_lik_obs::Matrix, log_A::Matrix)
    n_states = size(log_lik_obs, 2)
    n_obs = size(log_lik_obs, 1)
    xi = zeros(n_obs-1, n_states, n_states)

    log_gamma = log_f_obj.log_alpha .+ log_b_obj.log_beta
    log_gamma .-= logsumexp(log_gamma, dims=2)
    gamma = exp.(log_gamma)

    for t in 1:n_obs-1
        log_b_lik = log_lik_obs[t+1, :] .+ log_b_obj.log_beta[t+1, :]
        xi[t, :, :] .= (log_A .+ (log_f_obj.log_alpha[t, :] .+ log_b_lik')) .- log_f_obj.log_c[t+1]
    end

    xi = exp.(xi)
    xi_sum = drop_dim(sum(xi, dims=1))

    return (
        xi = xi_sum,
        gamma = gamma,
        pi0 = gamma[1, :]
    )
end
