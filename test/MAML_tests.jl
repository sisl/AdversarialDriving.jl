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

adapt(batch, policy, 0.1, baseline_weights, 1., λ)
outer_loss(batch, old_policy, new_policy, baseline_weights, γ, λ)

task_batch_loss(policy, task_sampler, N_tasks, N_eps_train, N_eps_test, inner_lr)

inner_loss(batch, policy)

adapt(batch, policy, inner_lr)

@test clip(10, 11, 100) == 11
@test clip(10, 1, 9) == 9
@test clip(10, 1, 11) == 10

outer_loss(batch, old_policy, new_policy)