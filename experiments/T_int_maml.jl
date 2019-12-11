include("plot_utils.jl")
include("../solver/MAML.jl")
include("../simulator/adm_task_generator.jl")

t = generate_ADM_POMDP()
policy = init_policy([o_dim(t), 150, 75, 25, a_dim(t)])

lr = .1
max_norm = 1.
training_log = Dict()
N_tasks = 8
N_eps_train = 150
N_eps_test = 150
inner_lr = 0.01

loss(p) = maml_task_batch_loss(p, gen_ADM_sampler(dt = 0.3, T=13), N_tasks, N_eps_train, N_eps_test, inner_lr, 0.99, 1., first_order = true, store_log = true, logger = training_log)

N_iterations = 50
filename = "T_int_MAML"
save_policy(filename, policy)
train_while_dropping_lr(policy, loss, N_iterations, lr, max_norm, training_log, filename)



plot_training(training_log)

