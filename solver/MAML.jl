include("ppo.jl")
using Flux.Tracker: gradient

function maml_task_batch_loss(policy, task_sampler, N_tasks, N_eps_train, N_eps_test, inner_lr, λ, baseline_reg_coeff; first_order = false, clip_ϵ = 0.2, store_log = false, logger = nothing)
    tasks = task_sampler(N_tasks) # return an array of pomdps
    total_loss = 0
    avg_ret_before, avg_max_ret_before = 0, 0
    avg_ret_after, avg_max_ret_after = 0, 0
    obs = Batch(N_tasks*max_steps(tasks[1])*N_eps_train, o_dim(tasks[1]), a_dim(tasks[1]))
    for task in tasks
        γ = discount(task)
        batch_before = sample_batch(task, policy, N_eps_train)
        if store_log
            append_ep!(obs, batch_before)
            avg_ret_before += average_episode_return(batch_before)
            avg_max_ret_before += max_episode_return(batch_before)
        end
        baseline_weights = fit(batch_before, baseline_reg_coeff)
        new_policy = adapt(batch_before, policy, inner_lr, baseline_weights, γ, λ, first_order)
        outer_logger = Dict()
        total_loss = total_loss + ppo_batch_loss(task, new_policy, N_eps_test, γ, λ, baseline_reg_coeff, ϵ = clip_ϵ, baseline_weights = baseline_weights, store_log = true, logger = outer_logger)
        if store_log
            avg_ret_after += outer_logger["return"][end]
            avg_max_ret_after += outer_logger["max_return"][end]
        end
    end
    loss = total_loss / N_tasks
    if store_log
        add_entry(logger, "loss", loss)
        logger["last_obs"] = obs.observations
        add_entry(logger, "return", avg_ret_after / N_tasks)
        add_entry(logger, "return_after", avg_ret_after / N_tasks)
        add_entry(logger, "max_return_after", avg_max_ret_after / N_tasks)
        add_entry(logger, "return_before", avg_ret_before / N_tasks)
        add_entry(logger, "max_return_before", avg_max_ret_before / N_tasks)
    end
    loss
end

# Computes the inner loss for the one-step gradient update. Uses basic REINFORCE with a baseline, computed with GAE
#NOTE: We are recomputing the mean and std dev of actions to get log_prob
#      this could probably be done during the rollouts
function inner_loss(batch, policy, baseline_weights, γ, λ)
    advantages = gae(batch, baseline_weights, γ, λ)
    log_probs = log_prob(batch.actions, forward_nn(policy, batch.observations)...)
    -mean(log_probs .* advantages)
end

# Update the policy by one gradient step of the inner loss
function adapt(batch, policy, inner_lr, baseline_weights, γ, λ, first_order)
    loss = inner_loss(batch, policy, baseline_weights, γ, λ)
    params = to_params(policy)
    grads = gradient(()->loss, params, nest = !first_order)

    max_norm = 1.0
    norm_p = 2.0
    scale = clip_norm_scale(grads, params, max_norm, norm_p)

    new_policy = MLPPolicy()
    for (k,p) in policy.weights
        if first_order
            new_policy.weights[k] = p - inner_lr*scale*grads[p].data
        else
            new_policy.weights[k] = p - inner_lr*scale*grads[p]
        end
    end
    new_policy
end

