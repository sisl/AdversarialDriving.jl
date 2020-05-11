include("../simulator/adm_task_generator.jl")
include("../solver/local_approx_policy_eval.jl")
include("plot_utils.jl")
include("../solver/is_probability_estimator.jl")
using Plots
pgfplots()
using POMDPSimulators
using POMDPPolicies
using Statistics
using Serialization
rng = MersenneTwister(0)

function run_trials(mdp, pol, Nsamps, Ntrials, rng)
    results = [sum([POMDPSimulators.simulate(RolloutSimulator(rng = rng), mdp, pol) for i=1:Nsamps]) for i=1:Ntrials]
    mean(results), std(results)
end

mdp = generate_2car_scene(dt = 0.18)

# Generate image of starting driving scenario
# p = plot_scene(mdp.initial_scene, mdp.models, mdp.roadway, egoid = mdp.egoid)

# Make gif of the nominal behavior of the scene
h_mc = POMDPSimulators.simulate(HistoryRecorder(rng = rng), mdp, FunctionPolicy((s) -> random_action(mdp, s, rng)))
# make_interact(state_hist(h_mc), mdp.models, mdp.roadway, egoid = mdp.egoid)
write_scenes(state_hist(h_mc), mdp.models, mdp.roadway, "two_car_nominal_frame", egoid = mdp.egoid)

########## Compare the policies #################
Ntrials = 5
Nsamples = 100

######### Monte Carlo Approach ################
mc_failures, mc_std = run_trials(mdp, FunctionPolicy((s) -> random_action(mdp, s, rng)), Nsamples, Ntrials, rng)
println("Monte-Carlo Rollouts failed: ", mc_failures, " ± ", mc_std, " / ", Nsamples)


######### Importance Sampling (uniform distribution) Approach ################
is_policy = UniformISPolicy(mdp, rng)
is_failures, is_std = run_trials(mdp, is_policy, Nsamples, Ntrials, rng)

## This is to output a vecotr of scenes
while true
    global hist = POMDPSimulators.simulate(HistoryRecorder(), mdp, is_policy)
    if undiscounted_reward(hist) > 0
        break
    end
end
undiscounted_reward(hist)
scenes = collect(state_hist(hist))
serialize("roadway.jls", mdp.roadway)
serialize("vector_of_scenes.jls", scenes)


# is_mean_arr, is_std_arr = compute_mean(50000, mdp, is_policy, rng)
println("Importance Sampling Rollouts failed: ", is_failures, " ± ", is_std, " / ", Nsamples)

############# Solve the problem w/ dynamic programming #########################
# N = 20
# veh = get_by_id(mdp.initial_scene, 1)
# ego = get_by_id(mdp.initial_scene, 2)
#
# grid = RectangleGrid(
# # Vehicle 1
#     range(posf(veh.state).s, stop=70,length=N), # position
#     range(0, stop=35., length=N), # Velocity
#     mdp.models[1].goals[laneid(veh)], # Goal
#     [0.0, 1.0], # Blinker
# # Vehicle 2 (ego)
#     range(posf(ego.state).s, stop=70, length=N), # position
#     range(0., stop=25., length=N), # Velocity
#     [5.0], # Goal
#     [1.0], # Blinker
#     )
#
# interp = LocalGIFunctionApproximator(grid)
# solver = LocalPolicyEvalSolver(interp, is_mdp_generative = true, n_generative_samples = 1, verbose = true, max_iterations = 100, belres = 1e-6)
# policy = solve(solver, mdp)
# serialize("two_car_policy.jls", policy)
policy = deserialize("two_car_policy.jls")


######### Local Approximation Policy Evaluation ############################
lape_failures, lape_std = run_trials(mdp, policy, Nsamples, Ntrials, rng)
# lape_mean_arr, lape_std_arr = compute_mean(50000, mdp, policy, rng)
println("Mean Utility Fusion Rollouts failed: ", lape_failures, " ± ", lape_std, " / ", Nsamples)


####### Plot the probability of failure convergence ##################
p = plot(title = "Probability of Failure", xlabel = "No. of Rollouts", ylabel = "Estimate of Probability of Failure")
plot!(is_mean_arr, ribbon = is_std_arr, label="Uniform Action Sampling")
plot!(lape_mean_arr, ribbon = lape_std_arr, label = "LAPE Sampling (ours)", xscale = :log)


##### Play some videos of the policies #######
while true
    global h = POMDPSimulators.simulate(HistoryRecorder(rng = rng), mdp, policy)
    if sum(reward_hist(h)) > 0
        break
    end
end
make_interact(state_hist(h), mdp.models, mdp.roadway, egoid = mdp.egoid)
write_scenes(state_hist(h), mdp.models, mdp.roadway, "two_car_collision_frame", egoid = mdp.egoid)

accelerations = [a[1].da for a in collect(action_hist(h))]
t = 0.18.*[0:length(accelerations)...]


using LaTeXStrings
plot([0, maximum(t)], [0, 0], linestyle = :dash, linecolor = :black, title = "Adversary Acceleration", ylabel = L"Acceleration (m/$s^2$)", xlabel = "Time (s)", label="", size = (600,200))
for i in 1:length(t)-1
    plot!([t[i], t[i+1]], [accelerations[i], accelerations[i]], label = "", linecolor = :black)
    if i < length(accelerations)
        plot!([t[i+1], t[i+1]], [accelerations[i], accelerations[i+1]], label = "", linecolor = :black)
    end
end

savefig("adversary_acc_twocar.pdf")

