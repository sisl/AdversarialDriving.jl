include("baseline.jl")
include("sampler.jl")
include("normal_mlp_policy.jl")

function task_batch_loss(policy, task_sampler, N_tasks, N_eps_train, N_eps_test, inner_lr, λ, baseline_reg_coeff)
    tasks = task_sampler(N_tasks) # return an array of pomdps
    for task in tasks
        γ = discount(task)
        batch_before = sample_batch(task, policy, N_eps_train)
        baseline_weights = fit(batch, baseline_reg_coeff)
        new_policy = adapt(batch_before, policy, inner_lr, baseline_weights, γ, λ)
        batch_after = sample_batch(task, new_policy, N_eps_test)
    end
    outer_loss(batch, policy, new_policy, baseline_weights, γ, λ)
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
function adapt(batch, policy, inner_lr, baseline_weights, γ, λ)
    loss = inner_loss(batch, policy, baseline_weights, γ, λ)
    grads = Tracker.gradient(()->loss, to_params(policy), nest = true)

    new_policy = Dict()
    for (k,p) in policy
        new_policy[k] = p - inner_lr*grads[p]
    end
    new_policy
end

# Clipping function
clip(val, low, high) = min(max(val, low), high)

# TODO: Also recomputing log probabilities here without reuse
function outer_loss(batch, old_policy, new_policy, baseline_weights, γ, λ, ϵ = 0.2)
    advantages = gae(batch, baseline_weights, γ, λ)
    if any(isinf.(advantages))
        println("found inf advantages")
    end

    old_log_probs = log_prob(batch.actions, forward_nn(old_policy, batch.observations)...)
    new_log_probs = log_prob(batch.actions, forward_nn(new_policy, batch.observations)...)
    ratio = exp.(new_log_probs .- old_log_probs)

    if any(isinf.(ratio))
        println("found inf ratio")
        println("ratio: ", clip.(ratio, 1-ϵ, 1+ϵ))
    end

    mean(min.(ratio.*advantages, clip.(ratio, 1-ϵ, 1+ϵ).*advantages))
end



