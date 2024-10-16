include("log_fb.jl");
incude("model_utils.jl");
include("data_log_likelihoods.jl");
include("m_step_functions.jl");

using LogExpFunctions;
using Distributions;
using LinearAlgebra;
using Random;
using Clustering;
using GLM;
using StatsBase;
using Optim;
using ProgressBars;
#using DoubleFloats;


function fit_glmhmm_with_em(
    X, 
    y, 
    n_states,
    dir_prior_diag,
    dir_prior_off_diag;
    model_init = nothing,
    rng_num = 9998,
    tol = 1e-3,
    max_iter = 150,
    n_class = 2
    )

    rng_state = MersenneTwister(rng_num)
    n_trials_total = size(X, 1)
    n_cols_bern = size(X, 2)

    if isnothing(model_init)
        b_init = init_bern(rng_state, X, y, n_states)
        model_init = (
            W = b_init
        )
    end

    W = model_init.W;


    dir_prior = generate_dir_prior(n_states, dir_prior_diag, dir_prior_off_diag)

    prob_obj = prob_param(
        init_A_from_prior(rng_state, n_states, dir_prior),
        [1 / n_states for n in 1:n_states],
        dir_prior
    )

    log_marg_lik = fill(NaN, max_iter+1)

    log_lik_obs = zeros(n_trials_total, n_states)
    
    compute_bern_loglikelihoods!(loglik_obs, X, y, W, n_trials_total, n_class, n_states)


    log_f_msg = log_forward(zeros(n_trials_total, n_states), zeros(n_trials_total))
    log_b_msg = log_backward(zeros(n_trials_total, n_states))

    log_forward_message!(log_f_msg, log_lik_obs, log.(prob_obj.pi0), prob_obj.A)
    log_backward_message!(log_b_msg, log_lik_obs, log.(prob_obj.A), log_f_msg.log_c)

    e_quants = expectations(log_f_msg, log_b_msg, log_lik_obs, prob_obj.A)
    prob_param_m_step!(prob_obj, e_quants)

    W = bern_m_step_with_derivs(X, y, e_quants.gamma, n_cols_bern, n_states)


    lml_prev = -Inf
    W_bern_old = copy(W_bern)
    converged = false
    iteration_counter = 1

    for iter in ProgressBar(1:max_iter)
        compute_bern_loglikelihoods!(loglik_obs, X, y, W, n_trials_total, n_class, n_states)

        log_forward_message!(log_f_msg, log_lik_obs, log.(prob_obj.pi0), log.(prob_obj.A))
        log_backward_message!(log_b_msg, log_lik_obs, log.(prob_obj.A), log_f_msg.log_c)

        e_quants = expectations(log_f_msg, log_b_msg, log_lik_obs, log.(prob_obj.A))
        prob_param_m_step!(prob_obj, e_quants)

        W_bern = bern_m_step_with_derivs(X, y, e_quants.gamma, n_cols_bern, n_states)

        lml = sum(log_f_msg.log_c)
        log_marg_lik[iter] = lml

        if abs(lml - lml_prev) < tol
            converged = true
            break
        end

        lml_prev = lml
        iteration_counter += 1
    end

    return (
        W = W,
        prob_obj = prob_obj,
        e_quants = e_quants,
        log_marg_lik = log_marg_lik[1:iteration_counter],
        converged = converged
    )
end
