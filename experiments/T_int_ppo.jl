include("../simulator/adm_task_generator.jl")
include("../solver/ppo.jl")
using Flux: ADAM
using Flux.Tracker: gradient
using LinearAlgebra, Plots

task = generate_ADM_POMDP(dt=0.3, T=13)
input_size, output_size = o_dim(task), a_dim(task)

policy = init_policy([input_size, 100, 50, 25, output_size], estimate_obs_stats = true)
params = to_params(policy)

opt = ADAM(0.001, (0.9, 0.999))
training_log = Dict()
loss() = ppo_batch_loss(task, policy, 700, 1., 0.95, 1., store_log = true, logger = training_log)

N, max_norm = 300, 1.
for i=1:1
    old_policy = deepcopy(policy)

    grads = gradient(() -> loss(), params)
    update_with_clip!(opt, grads, params, max_norm)

    add_entry(training_log, "kl", kl_divergence(policy, old_policy, training_log["last_obs"]))
    add_entry(training_log, "grad_norm", clipped_grad_norms(grads, params, max_norm))

    println("Finished epoch, ", i, " return: ", training_log["return"][end], " grad norms: ", training_log["grad_norm"][end])
end

plot_training(training_log)

# TODO: Replay episode

