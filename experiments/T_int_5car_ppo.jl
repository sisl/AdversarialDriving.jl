include("plot_utils.jl")
include("../simulator/adm_task_generator.jl")
include("../solver/ppo.jl")
using Flux: ADAM
using Flux.Tracker: gradient

filename = "T_int_5car_PPO"
task = generate_ADM_POMDP(dt=0.18, T=13)

# plot_scene(task.initial_scene, task.models, task.roadway; egoid = task.egoid)

policy = init_policy([o_dim(task), 100, 50, 25, a_dim(task)], σ_init = (in,out)->ones(in,out))

lr = .001
max_norm = 1.
training_log = Dict()
loss(p) = ppo_batch_loss(task, p, 1000, 1., 0.95, 1., store_log = true, logger = training_log)

N_iterations = 100
save_policy(filename, policy)
train!(policy, loss, N_iterations, lr, max_norm, training_log, filename)

plot_training(training_log,filename)




