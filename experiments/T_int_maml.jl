include("../solver/MAML.jl")
include("../simulator/adm_task_generator.jl")

t = generate_ADM_POMDP()
policy = init_policy([o_dim(t), 25, 25, 25, a_dim(t)])
[o_dim(t), 25, 25, 25, a_dim(t)]

lr = 0.001
max_norm = 1.
training_log = Dict()
N_tasks = 5
N_eps_train = 5
N_eps_test = 5
inner_lr = 0.1

loss(p) = maml_task_batch_loss(p, gen_ADM_sampler(dt = 0.3, T=13), N_tasks, N_eps_train, N_eps_test, inner_lr, 0.95, 1., first_order = true, store_log = true, logger = training_log)

N_iterations = 100
filename = "T_int_MAML"
save_policy(filename, policy)
train!(policy, loss, N_iterations, lr, max_norm, training_log, filename)

