include("ppo.jl")
using Flux.Tracker: gradient

function maml_task_batch_loss(policy, task_sampler, N_tasks, N_eps_train, N_eps_test, inner_lr, λ, baseline_reg_coeff, first_order = false, clip_ϵ = 0.2)
    tasks = task_sampler(N_tasks) # return an array of pomdps
    total_loss = 0
    for task in tasks
        γ = discount(task)
        batch_before = sample_batch(task, policy, N_eps_train)
        baseline_weights = fit(batch_before, baseline_reg_coeff)
        new_policy = adapt(batch_before, policy, inner_lr, baseline_weights, γ, λ, first_order)
        total_loss = total_loss + ppo_batch_loss(task, new_policy, N_eps_test, γ, λ, baseline_reg_coeff, ϵ = clip_ϵ, baseline_weights = baseline_weights)
    end
    total_loss / N_tasks
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
    grads = gradient(()->loss, to_params(policy), nest = !first_order)

    new_policy = MLPPolicy()
    new_policy.input_mean = policy.input_mean
    new_policy.input_std = policy.input_std
    for (k,p) in policy.weights
        if first_order
            new_policy.weights[k] = p - inner_lr*grads[p].data
        else
            new_policy.weights[k] = p - inner_lr*grads[p]
        end
    end
    new_policy
end

