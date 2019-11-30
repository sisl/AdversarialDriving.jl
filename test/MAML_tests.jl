include("../simulator/adm_task_generator.jl")
include("../solver/MAML.jl")
using Test

t = generate_ADM_POMDP()
γ = discount(t)
λ = 0.5
s0 = initialstate(t)
input_size, output_size = o_dim(t), a_dim(t)
policy = init_policy([input_size,10,output_size])

# Sample data from a task
batch = sample_batch(t, policy, 3)
baseline_weights = fit(batch, 1000.)

inner_loss(batch, policy, baseline_weights, 1., 1.)

new_policy = adapt(batch, policy, 1e-6, baseline_weights, 1., λ)
outer_loss(batch, policy, new_policy, baseline_weights, γ, λ)


task_batch_loss(policy, sample_ADM_POMDPs, 4, 10, 3, 1e-6, 0.5, 1.)

@test clip(10, 11, 100) == 11
@test clip(10, 1, 9) == 9
@test clip(10, 1, 11) == 10
