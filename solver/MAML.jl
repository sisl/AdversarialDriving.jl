function task_batch_loss(policy, task_sampler, N_tasks, N_eps_train, N_eps_test, inner_lr)
    tasks = task_sampler(N_tasks) # return an array of pomdps
    for task in tasks
        batch_before = sample(task, policy, N_eps_train)
        new_policy = adapt(batch_before, policy, inner_lr)
        batch_after = sample(task, new_policy, N_eps_test)
    end
    outer_loss(batch, policy, new_policy)
end

# Computes the inner loss for the one-step gradient update. Uses basic REINFORCE with a baseline, computed with GAE
#NOTE: We are recomputing the mean and std dev of actions to get log_prob
#      this could probably be done during the rollouts
function inner_loss(batch, policy)
    advantages = gae(batch)
    log_probs = log_prob(batch.actions, forward_nn(policy, batch.observations))
    -mean(log_probs .* advantages)
end

function adapt(batch, policy, inner_lr)
    loss = inner_loss(batch, policy)
    grads = Tracker.gradient(()->loss, to_params(policy), nest = true)

    new_policy = Dict()
    for (k,p) in policy
        new_policy[k] = p - inn_lr*grads[p]
    end
    new_policy
end

# TODO: Also recomputing log probabilities here without reuse
function outer_loss(batch, old_policy, new_policy)
    advantages = gae(batch)
    old_log_probs = log_prob(batch.actions, forward_nn(old_policy, batch.observations))
    new_log_probs = log_prob(batch.actions, forward_nn(new_policy, batch.observations))
    log_ratio = (new_log_probs .- old_log_probs)
    ratio = torch.exp(log_ratio)
    #TODO: Clip ratio according to PPO
    -mean(ratio * advantages)
end



