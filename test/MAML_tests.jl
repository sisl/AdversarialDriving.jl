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

new_policy = adapt(batch, policy, 1e-6, baseline_weights, 1., λ, false)
new_policy = adapt(batch, policy, 1e-6, baseline_weights, 1., λ, true)


maml_task_batch_loss(policy = policy, task_sampler = sample_ADM_POMDPs, N_tasks = 5, N_eps_train = 10, N_eps_test = 3, inner_lr = 1e-6, λ = 0.5, baseline_reg_coeff = 1.)

@test clamp(10, 11, 100) == 11
@test clamp(10, 1, 9) == 9
@test clamp(10, 1, 11) == 10

