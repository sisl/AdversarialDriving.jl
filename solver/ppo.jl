include("baseline.jl")
include("sampler.jl")
include("normal_mlp_policy.jl")
include("optimization.jl")


# Surrogate loss for PPO
function ppo_batch_loss(task, policy, N_eps, γ, λ, baseline_reg_coeff; ϵ = 0.2, baseline_weights = nothing, store_log = false, logger = nothing)
    # Sample a batch from the current policy
    batch = sample_batch(task, policy, N_eps)
    store_log && add_entry(logger, "return", average_episode_return(batch))
    store_log && add_entry(logger, "max_return", max_episode_return(batch))
    store_log && (logger["last_obs"] = batch.observations)

    # Compute advantages with linear baseline
    isnothing(baseline_weights) && (baseline_weights = fit(batch, baseline_reg_coeff))
    advantages = gae(batch, baseline_weights, γ, λ)

    # Compute the log ratio fro the derivative and use it for the loss
    old_log_probs = log_prob(batch.actions, forward_nn(policy, batch.observations)...).data
    new_log_probs = log_prob(batch.actions, forward_nn(policy, batch.observations)...)
    ratio = exp.(new_log_probs .- old_log_probs)
    loss = -mean(min.(ratio.*advantages, clamp.(ratio, 1-ϵ, 1+ϵ).*advantages))
    store_log && add_entry(logger, "loss", loss)
    loss
end

