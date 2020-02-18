include("../simulator/adm_task_generator.jl")
include("../solver/local_approx_policy_eval.jl")
include("plot_utils.jl")
using Plots
using POMDPSimulators
using POMDPPolicies
using Statistics
rng = MersenneTwister(0)

function run_trials(mdp, pol, Nsamps, Ntrials, rng)
    results = [sum([POMDPSimulators.simulate(RolloutSimulator(rng = rng), mdp, pol) for i=1:Nsamps]) for i=1:Ntrials]
    mean(results), std(results)
end

mdp = generate_2car_scene(dt = 0.18)

# Generate image of starting driving scenario
p = plot_scene(mdp.initial_scene, mdp.models, mdp.roadway, egoid = mdp.egoid)
write_to_svg(p, "twocar_scenario.svg")

# Make gif of the nominal behavior of the scene
h_mc = POMDPSimulators.simulate(HistoryRecorder(rng = rng), mdp, FunctionPolicy((s) -> random_action(mdp, s, rng)))
make_interact(state_hist(h_mc), mdp.models, mdp.roadway, egoid = mdp.egoid)

write_scenes(state_hist(h_mc), mdp.models, mdp.roadway, "frame", egoid = mdp.egoid)

########## Compare the policies #################
Ntrials = 5
Nsamples = 100

######### Monte Carlo Approach ################
mc_failures, mc_std = run_trials(mdp, FunctionPolicy((s) -> random_action(mdp, s, rng)), Nsamples, Ntrials, rng)
println("Monte-Carlo Rollouts failed: ", mc_failures, " ± ", mc_std, " / ", Nsamples)

######### Importance Sampling (uniform distribution) Approach ################
is_failures, is_std = run_trials(mdp, FunctionPolicy((s) -> rand(rng, actions(mdp, s))), Nsamples, Ntrials, rng)
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
policy = deserialize("two_car_failure_pol.jls")


######### Local Approximation Policy Evaluation ############################
lape_failures, lape_std = run_trials(mdp, policy, Nsamples, Ntrials, rng)
println("Mean Utility Fusion Rollouts failed: ", lape_failures, " ± ", lape_std, " / ", Nsamples)

##### Play some videos of the policies #######
h_min_ga = POMDPSimulators.simulate(HistoryRecorder(rng = rng), mdp, policy)
make_interact(state_hist(h_min_ga), mdp.models, mdp.roadway, egoid = mdp.egoid)

write_scenes(state_hist(h_min_ga), mdp.models, mdp.roadway, "frame", egoid = mdp.egoid)

