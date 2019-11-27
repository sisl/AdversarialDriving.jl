include("../solver/sampler.jl")
using Test

#NOTE: The "sample" function is in MAML_test because it requires a POMDP

# Batch constructor
N, o, a = 100, 3, 6
batch = Batch(N, o, a)
# Test that the correct array sizes are constructed
@test size(batch.observations) == (N,o)
@test size(batch.actions) == (N,a)
@test size(batch.returns) == (N,)
@test size(batch.times) == (N,)
@test size(batch.rewards) == (N,)
@test size(batch.episode_ends) == (N,)

# Check that episode ends are initialized to false
@test all(batch.episode_ends .== false)

# Check that the counter of the number of entries starts at 0
@test batch.N == 0

# create a fake episode
N_step = 10
obs, acts = ones(N_step,o), ones(N_step,a)
times = cumsum(ones(N_step))
rewards = ones(N_step)
rets = returns(rewards, 0.9)

# Check that the returns are computed appropriately
@test all(isapprox.(rets, [(1-0.9^(N_step-i+1)) / (1-0.9) for i in 1:N_step]))

# Check for size assertions in append_ep!:
@test_throws AssertionError append_ep!(batch, ones(N_step+1, o), acts, rewards, rets, times)
@test_throws AssertionError append_ep!(batch, ones(N_step, o+1), acts, rewards, rets, times)
@test_throws AssertionError append_ep!(batch, obs, ones(N_step+1, a), rewards, rets, times)
@test_throws AssertionError append_ep!(batch, obs, ones(N_step, a+1), rewards, rets, times)
@test_throws AssertionError append_ep!(batch, obs, acts, ones(N_step-1), rets, times)
@test_throws AssertionError append_ep!(batch, obs, acts, rewards, ones(N_step-1), times)
@test_throws AssertionError append_ep!(batch, obs, acts, rewards, rets, ones(N_step-1))

# Check for inserting too large an episode
small_batch = Batch(3, o, a)
@test_throws AssertionError append_ep!(small_batch, obs, acts, rewards, rets, times)

# Check for inserting too many episodes
fill_batch = Batch(15, o, a)
append_ep!(fill_batch, obs, acts, rewards, rets, times)
@test_throws AssertionError append_ep!(fill_batch, obs, acts, rewards, rets, times)

# append the appropriate stuff to the batch
append_ep!(batch, obs, acts, rewards, rets, times)

# Check that the epsiode was appended and the counter was incremented
@test batch.observations[1:N_step, :] == obs
@test batch.actions[1:N_step, :] == acts
@test batch.rewards[1:N_step] == rewards
@test batch.returns[1:N_step] == rets
@test batch.episode_ends[N_step] == true
@test batch.N == N_step

# Trim the batch
trimmed_batch = trim(batch)
@test trimmed_batch.N == N_step
@test trimmed_batch.observations == obs
@test trimmed_batch.actions == acts
@test trimmed_batch.rewards == rewards
@test trimmed_batch.returns == rets
@test trimmed_batch.episode_ends[end] == true




