include("plot_utils.jl")
include("../simulator/gym_continous.jl")
include("../solver/ppo.jl")

env = make("Pendulum-v0", :human_pane)
policy = init_policy([o_dim(env), 100, 50, 25, a_dim(env)])


lr = 0.001
max_norm = 1
training_log = Dict()
loss(p) = ppo_batch_loss(env, p, 125, 1., 0.95, 1., store_log = true, logger = training_log)

N_iterations = 100
filename = "pendulum_ppo"
save_policy(filename, policy)
train!(policy, loss, N_iterations, lr, max_norm, training_log, filename)
# train_with_restarts(policy, loss, N_iterations, lr, max_norm, training_log, "pendulum_ppo.model")

plot_training(training_log)


i, done, o, tot_r = 1, false, reset!(env), 0
theta, dtheta = [], []
while !done
    a = sample_action(policy, o)
    global o, r, done, d = step!(env, a)
    global tot_r += r
    push!(theta, env._env.state[1])
    push!(dtheta, env._env.state[2])
    global i += 1
    (i > env.max_episode_steps) && break
end
println("total reward: ", tot_r)

scatter(theta, dtheta, xlabel="theta", ylabel="dtheta")

