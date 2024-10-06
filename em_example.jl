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
    X_bern, 
    X_gauss, 
    y_bern, 
    Y_gauss, 
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
    n_trials_total = size(X_bern, 1)
    n_cols_bern = size(X_bern, 2)
    n_cols_gauss = size(X_gauss[1], 2)
    n_dep_var = size(Y_gauss[1], 2)
    n_samp_per_trial = size.(Y_gauss, 1)

    big_x_gauss = vcat(X_gauss...)
    big_y_gauss = vcat(Y_gauss...)

    if isnothing(model_init)
        g_init = init_gauss(rng_state, big_x_gauss, big_y_gauss, n_states)
        b_init = init_bern(rng_state, X_bern, y_bern, n_states)
        model_init = (
            W_bern = b_init,
            W_gauss = g_init.W,
            sig2 = g_init.sig2
        )
    end

    W_bern = model_init.W_bern;
    W_gauss = model_init.W_gauss;
    sig2 = model_init.sig2;

    dir_prior = generate_dir_prior(n_states, dir_prior_diag, dir_prior_off_diag)

    prob_obj = prob_param(
        init_A_from_prior(rng_state, n_states, dir_prior),
        [1 / n_states for n in 1:n_states],
        dir_prior
    )

    log_marg_lik = fill(NaN, max_iter+1)

    # Initialize log-likelihood matrices
    loglik_obs_bern = zeros(n_trials_total, n_states)
    loglik_obs_gauss = zeros(n_trials_total, n_states)
    log_lik_obs = zeros(n_trials_total, n_states)
    
    compute_bern_loglikelihoods!(loglik_obs_bern, X_bern, y_bern, W_bern, n_trials_total, n_class, n_states)
    compute_gauss_loglikelihoods!(loglik_obs_gauss, X_gauss, Y_gauss, W_gauss, sig2, n_trials_total, n_states)
    log_lik_obs .= loglik_obs_bern .+ loglik_obs_gauss

    log_f_msg = log_forward(zeros(n_trials_total, n_states), zeros(n_trials_total))
    log_b_msg = log_backward(zeros(n_trials_total, n_states))

    log_forward_message!(log_f_msg, log_lik_obs, log.(prob_obj.pi0), prob_obj.A)
    log_backward_message!(log_b_msg, log_lik_obs, log.(prob_obj.A), log_f_msg.log_c)

    e_quants = expectations(log_f_msg, log_b_msg, log_lik_obs, prob_obj.A)
    prob_param_m_step!(prob_obj, e_quants)

    W_bern = bern_m_step_with_derivs(X_bern, y_bern, e_quants.gamma, n_cols_bern, n_states)
    W_gauss, sig2 = gauss_m_step(big_x_gauss, big_y_gauss, e_quants.gamma, n_samp_per_trial)


    lml_prev = -Inf
    W_bern_old = copy(W_bern)
    converged = false
    iteration_counter = 1

    for iter in ProgressBar(1:max_iter)
        compute_bern_loglikelihoods!(loglik_obs_bern, X_bern, y_bern, W_bern, n_trials_total, n_class, n_states)
        compute_gauss_loglikelihoods!(loglik_obs_gauss, X_gauss, Y_gauss, W_gauss, sig2,  n_trials_total, n_states)
        log_lik_obs .= loglik_obs_bern .+ loglik_obs_gauss

        log_forward_message!(log_f_msg, log_lik_obs, log.(prob_obj.pi0), log.(prob_obj.A))
        log_backward_message!(log_b_msg, log_lik_obs, log.(prob_obj.A), log_f_msg.log_c)

        e_quants = expectations(log_f_msg, log_b_msg, log_lik_obs, log.(prob_obj.A))
        prob_param_m_step!(prob_obj, e_quants)

        W_bern = bern_m_step_with_derivs(X_bern, y_bern, e_quants.gamma, n_cols_bern, n_states)
        W_gauss, sig2 = gauss_m_step(big_x_gauss, big_y_gauss, e_quants.gamma, n_samp_per_trial)

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
        W_bern = W_bern,
        W_gauss = W_gauss,
        sig2 = sig2,
        prob_obj = prob_obj,
        e_quants = e_quants,
        log_marg_lik = log_marg_lik[1:iteration_counter],
        converged = converged
    )
end
