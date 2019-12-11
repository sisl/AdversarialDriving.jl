include("plot_utils.jl")
include("../simulator/adm_task_generator.jl")
include("../solver/ppo.jl")
using Flux: ADAM
using Flux.Tracker: gradient

task = generate_ADM_POMDP(dt=0.3, T=13)

policy = init_policy([o_dim(task), 100, 50, 25, a_dim(task)], σ_init = (in,out)->0.01*ones(in,out))

lr = 0.001
max_norm = 1.
training_log = Dict()
loss(p) = ppo_batch_loss(task, p, 550, 1., 0.95, 1., store_log = true, logger = training_log)

N_iterations = 500
filename = "T_int_PPO"
save_policy(filename, policy)
# train_with_restarts(policy, loss, N_iterations, lr, max_norm, training_log, "T_int.model")
train!(policy, loss, N_iterations, lr, max_norm, training_log, filename)

plot_training(training_log)

# TODO: Replay episode

