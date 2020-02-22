include("../simulator/adm_task_generator.jl")
include("../solver/linear_mc_policy_eval.jl")
include("../solver/local_approx_policy_eval.jl")
include("plot_utils.jl")
include("../solver/is_probability_estimator.jl")
using Serialization
using Plots
using POMDPSimulators
using POMDPPolicies
using Statistics

rng = MersenneTwister(0)

function run_trials(mdp, pol, Nsamps, Ntrials, rng)
    results = [sum([POMDPSimulators.simulate(RolloutSimulator(rng = rng), mdp, pol) for i=1:Nsamps]) for i=1:Ntrials]
    mean(results), std(results)
end

decomposed, mdp = generate_decomposed_scene(dt = 0.18)

# Generate image of starting driving scenario
# p = plot_scene(mdp.initial_scene, mdp.models, mdp.roadway, egoid = mdp.egoid)

# Generate a gif of the nominal behavior of the 5-car scenario
h_mc = POMDPSimulators.simulate(HistoryRecorder(rng = rng), mdp, FunctionPolicy((s) -> actions(mdp)[1]))
# make_interact(state_hist(h_mc), mdp.models, mdp.roadway, egoid = mdp.egoid)
write_scenes(state_hist(h_mc), mdp.models, mdp.roadway, "fivecar_nominal_frame", egoid = mdp.egoid)


Ns = length(convert_s_expanded(Vector{Float64},  initialstate(mdp), mdp))
Ntrials = 5
Nsamples = 100


######## Load in policies ################
policies = [deserialize(string("policy_decomp_", i, ".jls")) for i in 1:length(decomposed)]
Vest_mean = subproblem_estimate_fn(policies, (s) -> decompose_scene(s, mdp.egoid), :mean)
Vest_min = subproblem_estimate_fn(policies, (s) -> decompose_scene(s, mdp.egoid), :min)
Vest_max = subproblem_estimate_fn(policies, (s) -> decompose_scene(s, mdp.egoid), :max)
is_policy_no_estimate = ISPolicy(mdp, LinearModel(Ns), (s) -> 0, convert_s_expanded, :all)
is_policy_subprob_mean = ISPolicy(mdp, LinearModel(Ns), Vest_mean, convert_s_expanded, :all)
is_policy_subprob_min = ISPolicy(mdp, LinearModel(Ns), Vest_min, convert_s_expanded, :all)
is_policy_subprob_max = ISPolicy(mdp, LinearModel(Ns), Vest_max, convert_s_expanded, :all)

##### Generate set of states that will be used for bellman residual ##########
# Nstates = 100
# Bellman_states = []
#
# while length(Bellman_states) < Nstates
#     history = POMDPSimulators.simulate(HistoryRecorder(rng = rng), mdp, is_policy_subprob_mean)
#     if history[end].r == 1
#         for s in state_hist(history)
#             push!(Bellman_states, s)
#             length(Bellman_states) >= Nstates && break
#         end
#     end
# end


######### Monte Carlo Approach ################
mc_failures, mc_std = run_trials(mdp, FunctionPolicy((s) -> random_action(mdp, s, rng)), Nsamples, Ntrials, rng)
println("Monte-Carlo Rollouts failed: ", mc_failures, " ± ", mc_std, " / ", Nsamples)

######### Importance Sampling (uniform distribution) Approach ################
is_policy = UniformISPolicy(mdp, rng)
is_failures, is_std = run_trials(mdp, is_policy, Nsamples, Ntrials, rng)
# is_mean_arr, is_std_arr = compute_mean(50000, mdp, policy, rng)
# zero_guess_residuals = bellman_residual(mdp, Bellman_states, (s)->0)
println("Importance Sampling Rollouts failed: ", is_failures, " ± ", is_std, " / ", Nsamples)#, " Bellman Residual: max=", maximum(zero_guess_residuals), " mean=", mean(zero_guess_residuals))

######### Subproblem Estimation Approaches (No global Approximation) ##########

# Utility Fusion with Mean
mean_uf_failures, mean_uf_std = run_trials(mdp, is_policy_subprob_mean, Nsamples, Ntrials, rng)
# mean_uf_mean_arr, mean_uf_std_arr = compute_mean(50000, mdp, is_policy_subprob_mean, rng)
# mean_uf_residuals = bellman_residual(mdp, Bellman_states, (s) -> value(is_policy_subprob_mean, s))
println("Mean Utility Fusion Rollouts failed: ", mean_uf_failures, " ± ", mean_uf_std, " / ", Nsamples)#,  " Bellman Residual: max=", maximum(mean_uf_residuals), " mean=", mean(mean_uf_residuals))


# Utility Fusion with Max
max_uf_failures, max_uf_std = run_trials(mdp, is_policy_subprob_max, Nsamples, Ntrials, rng)
# max_uf_mean_arr, max_uf_std_arr = compute_mean(50000, mdp, is_policy_subprob_max, rng)
# max_uf_residuals = bellman_residual(mdp, Bellman_states, (s) -> value(is_policy_subprob_max, s))
println("Max Utility Fusion Rollouts failed: ", max_uf_failures, " ± ", max_uf_std, " / ", Nsamples)#,  " Bellman Residual: max=", maximum(max_uf_residuals), " mean=", mean(max_uf_residuals))


# Utility Fusion with Min
min_uf_failures, min_uf_std = run_trials(mdp, is_policy_subprob_min, Nsamples, Ntrials, rng)
# min_uf_mean_arr, min_uf_std_arr = compute_mean(50000, mdp, is_policy_subprob_min, rng)
# min_uf_residuals = bellman_residual(mdp, Bellman_states, (s) -> value(is_policy_subprob_min, s))
println("Min Utility Fusion Rollouts failed: ", min_uf_failures, " ± ", min_uf_std, " / ", Nsamples)#,  " Bellman Residual: max=", maximum(min_uf_residuals), " mean=", mean(min_uf_residuals))


######### Train the global approximation ##########
iterations = 25
eps_per_iteration = 100

######### Subproblem Estimation Approaches (No global Approximation) ##########
# No Estimation -- Just global approximation trained with IS
println("Training Global Approx Model (No Estimate)... ")
is_failure_rate = []
mc_policy_eval!(is_policy_no_estimate, iterations, eps_per_iteration, verbose = false, failure_rate_vec = is_failure_rate)
no_est_ga_failures, no_est_ga_std = run_trials(mdp, is_policy_no_estimate, Nsamples, Ntrials, rng)
# no_est_ga_residuals = bellman_residual(mdp, Bellman_states, (s) -> value(is_policy_no_estimate, s))
println("No Estimate - Global Approximation Failed: ", no_est_ga_failures, " ± ", no_est_ga_std, " / ", Nsamples)#,  " Bellman Residual: max=", maximum(no_est_ga_residuals), " mean=", mean(no_est_ga_residuals))
serialize("is_failure_rate.jls", is_failure_rate)

# Utility Fusion with Mean + Global Approximation
println("Training Global Approx Model (Mean UF)... ")
mean_uf_failure_rate = []
mc_policy_eval!(is_policy_subprob_mean, iterations, eps_per_iteration, verbose = false, failure_rate_vec = mean_uf_failure_rate)
mean_uf_ga_failures, mean_uf_ga_std = run_trials(mdp, is_policy_subprob_mean, Nsamples, Ntrials, rng)
# mean_uf_ga_residuals = bellman_residual(mdp, Bellman_states, (s) -> value(is_policy_subprob_mean, s))
println("Mean Utility Fusion Rollouts failed: ", mean_uf_ga_failures, " ± ", mean_uf_ga_std, " / ", Nsamples)#,  " Bellman Residual: max=", maximum(mean_uf_ga_residuals), " mean=", mean(mean_uf_ga_residuals))
serialize("mean_uf_failure_rate.jls", mean_uf_failure_rate)

# Utility Fusion with Max + Global Approximation
println("Training Global Approx Model (Max UF)... ")
max_uf_failure_rate = []
mc_policy_eval!(is_policy_subprob_max, iterations, eps_per_iteration, verbose = false, failure_rate_vec = max_uf_failure_rate)
max_uf_ga_failures, max_uf_ga_std = run_trials(mdp, is_policy_subprob_max, Nsamples, Ntrials, rng)
# max_uf_ga_residuals = bellman_residual(mdp, Bellman_states, (s) -> value(is_policy_subprob_max, s))
println("Max Utility Fusion Rollouts failed: ", max_uf_ga_failures, " ± ", max_uf_ga_std, " / ", Nsamples)#,  " Bellman Residual: max=", maximum(max_uf_ga_residuals), " mean=", mean(max_uf_ga_residuals))
serialize("max_uf_failure_rate.jls", max_uf_failure_rate)

# Utility Fusion with Min + Global Approximation
println("Training Global Approx Model (Min UF)... ")
min_uf_failure_rate = []
mc_policy_eval!(is_policy_subprob_min, iterations, eps_per_iteration, verbose = false, failure_rate_vec = min_uf_failure_rate)
min_uf_ga_failures, min_uf_ga_std = run_trials(mdp, is_policy_subprob_min, Nsamples, Ntrials, rng)
# min_uf_ga_residuals = bellman_residual(mdp, Bellman_states, (s) -> value(is_policy_subprob_min, s))
println("Min Utility Fusion Rollouts failed: ", min_uf_ga_failures, " ± ", min_uf_ga_std, " / ", Nsamples)#,  " Bellman Residual: max=", maximum(min_uf_ga_residuals), " mean=", mean(min_uf_ga_residuals))
serialize("min_uf_failure_rate.jls", min_uf_failure_rate)

pgfplots()
plot(is_failure_rate, title = "Failure Rate Training Curves", label = "GLA", xlabel = string("Iterations (", eps_per_iteration, " Episodes per)"), ylabel = "Failure Rate", linewidth = 2)
plot!(mean_uf_failure_rate, label = "Mean UF + GLA", linewidth = 2)
plot!(max_uf_failure_rate, label = "Max UF + GLA", linewidth = 2)
plot!(min_uf_failure_rate, label = "Min UF + GLA", linewidth = 2)
savefig("learning_curv.pdf")


##### Playback a failure and play frames #######
while true
    global h_min_ga = POMDPSimulators.simulate(HistoryRecorder(rng = rng), mdp, is_policy_subprob_mean)
    if sum(reward_hist(h_min_ga)) > 0
        break
    end
end
# make_interact(state_hist(h_min_ga), mdp.models, mdp.roadway, egoid = mdp.egoid)
write_scenes(state_hist(h_min_ga), mdp.models, mdp.roadway, "five_car_collision_frame", egoid = mdp.egoid)

