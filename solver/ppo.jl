include("baseline.jl")
include("sampler.jl")
include("normal_mlp_policy.jl")
include("optimization.jl")


# Surrogate loss for PPO
function ppo_batch_loss(task, policy, N_eps, γ, λ, baseline_reg_coeff; ϵ = 0.2, baseline_weights = nothing)
    batch = sample_batch(task, policy, N_eps)
    isnothing(baseline_weights) && (baseline_weights = fit(batch, baseline_reg_coeff))
    advantages = gae(batch, baseline_weights, γ, λ)

    old_log_probs = log_prob(batch.actions, forward_nn(policy, batch.observations)...).data
    new_log_probs = log_prob(batch.actions, forward_nn(policy, batch.observations)...)
    ratio = exp.(new_log_probs .- old_log_probs)

    -mean(min.(ratio.*advantages, clamp.(ratio, 1-ϵ, 1+ϵ).*advantages))
end

