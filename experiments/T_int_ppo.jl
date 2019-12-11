include("plot_utils.jl")
include("../simulator/adm_task_generator.jl")
include("../solver/ppo.jl")
using Flux: ADAM
using Flux.Tracker: gradient

task = generate_ADM_POMDP(dt=0.3, T=13)

policy = init_policy([o_dim(task), 100, 50, a_dim(task)], σ_init = (in,out)->0.2*ones(in,out))

lr = .1
max_norm = 1.
training_log = Dict()
loss(p) = ppo_batch_loss(task, p, 1000, 1., 0.99, 1., store_log = true, logger = training_log)

N_iterations = 50
filename = "T_int_PPO"
save_policy(filename, policy)
train_while_dropping_lr(policy, loss, N_iterations, lr, max_norm, training_log, filename)
# train!(policy, loss, N_iterations, lr, max_norm, training_log, filename)

plot_training(training_log)

policy = load_policy("T_int_PPO")
batch = sample_batch(task, policy, 1)
histogram(batch.actions[3:3:end])

# TODO: Replay episode

