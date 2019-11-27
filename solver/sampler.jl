using Parameters
using Random

# Stores the batch information for a series of episodes for a given task and policy
mutable struct Batch
    observations::Array{Float64, 2}
    actions::Array{Float64, 2}
    rewards::Array{Float64}
    returns::Array{Float64}
    times::Array{Float64}
    episode_ends::Array{Bool}
    N::Int64
end

# Construct a batch with pre-allocated memory
function Batch(N_max, o_dim, a_dim)
    Batch(Array{Float64,2}(undef, N_max, o_dim), # observations
          Array{Float64,2}(undef, N_max, a_dim), # actions
          Array{Float64}(undef, N_max), # rewards
          Array{Float64}(undef, N_max), # returns
          Array{Float64}(undef, N_max), # times
          fill(false, N_max), # episode ends
          0 # size of contents
          )
end

# Append an episode's worth of data to the Batch
function append_ep!(batch::Batch, obs, as, rs, rets, times)
    @assert size(obs,1) == size(as,1) && size(as,1) == length(rs) && length(rs) == length(rets) && length(rets) == length(times) && size(obs, 2) == size(batch.observations, 2) && size(as, 2) == size(batch.actions, 2)

    N_new = size(as, 1)
    @assert N_new + batch.N <= size(batch.actions, 1) # Make sure we don't overrun the arrays

    batch.observations[batch.N + 1 : batch.N + N_new, :] .= obs
    batch.actions[batch.N + 1 : batch.N + N_new, :] .= as
    batch.rewards[batch.N + 1 : batch.N + N_new] .= rs
    batch.returns[batch.N + 1 : batch.N + N_new] .= rets
    batch.times[batch.N + 1 : batch.N + N_new] .= times
    batch.episode_ends[batch.N + N_new] = true
    batch.N += N_new
end

function trim(batch::Batch)
    Batch(batch.observations[1:batch.N, :],
          batch.actions[1:batch.N, :],
          batch.rewards[1:batch.N],
          batch.returns[1:batch.N],
          batch.times[1:batch.N],
          batch.episode_ends[1:batch.N],
          batch.N)
end

# Compute the returns from the rewards
function returns(rs, γ)
    returns = zeros(length(rs))
    _return = 0
    for i=length(returns):-1:1
        _return = _return*γ + rs[i]
        returns[i] = _return
    end
    returns
end

# Sample a given number of episodes from a task given a policy
# Returns the results as a batch
function sample_batch(task, policy, N_eps; rng::AbstractRNG = Random.GLOBAL_RNG)
    batch = Batch(max_steps(task)*N_eps, o_dim(task), a_dim(task))
    for i = 1:N_eps
        s0 = initialstate(task)
        obs, as, rs = policy_rollout(pomdp, (obs) -> sample_action(policy, obs, rng=rng), s0)
        rets = returns(rs, discount(pomdp))
        times = range(0, length=length(rs), step=task.dt)
        append_ep!(batch, obs, as, rs, rets, times)
    end
    trim(batch)
end

