include("../simulator/adm_task_generator.jl")
include("../solver/linear_mc_policy_eval.jl")
include("../solver/local_approx_policy_eval.jl")
include("plot_utils.jl")
using Serialization
using Plots
using POMDPSimulators
using POMDPPolicies
using Statistics

function run_trials(mdp, pol, Nsamps, Ntrials, rng)
    results = [sum([POMDPSimulators.simulate(RolloutSimulator(rng = rng), mdp, pol) for i=1:Nsamps]) for i=1:Ntrials]
    mean(results), std(results)
end

decomposed, mdp = generate_decomposed_scene(dt = 0.18)
Ns = length(convert_s_expanded(Vector{Float64},  initialstate(mdp), mdp))
Nactions = 100
Ntrials = 5
Nsamples = 100
rng = MersenneTwister(0)

######## Load in policies ################
policies = [deserialize(string("policy_decomp_", i, ".jls")) for i in 1:length(decomposed)]
Vest_mean = subproblem_estimate_fn(policies, (s) -> decompose_scene(s, mdp.egoid), :mean)
Vest_min = subproblem_estimate_fn(policies, (s) -> decompose_scene(s, mdp.egoid), :min)
Vest_max = subproblem_estimate_fn(policies, (s) -> decompose_scene(s, mdp.egoid), :max)
is_policy_no_estimate = ISPolicy(mdp, LinearModel(Ns), (s) -> 0, convert_s_expanded, Nactions)
is_policy_subprob_mean = ISPolicy(mdp, LinearModel(Ns), Vest_mean, convert_s_expanded, Nactions)
is_policy_subprob_min = ISPolicy(mdp, LinearModel(Ns), Vest_min, convert_s_expanded, Nactions)
is_policy_subprob_max = ISPolicy(mdp, LinearModel(Ns), Vest_max, convert_s_expanded, Nactions)

##### Generate set of states that will be used for bellman residual ##########
Nstates = 100
Bellman_states = []

while length(Bellman_states) < Nstates
    history = POMDPSimulators.simulate(HistoryRecorder(rng = rng), mdp, is_policy_subprob_mean)
    if history[end].r == 1
        for s in state_hist(history)
            push!(Bellman_states, s)
            length(Bellman_states) >= Nstates && break
        end
    end
end


######### Monte Carlo Approach ################
mc_failures, mc_std = run_trials(mdp, FunctionPolicy((s) -> random_action(mdp, s, rng)), Nsamples, Ntrials, rng)
println("Monte-Carlo Rollouts failed: ", mc_failures, " ± ", mc_std, " / ", Nsamples)

######### Importance Sampling (uniform distribution) Approach ################
is_failures, is_std = run_trials(mdp, FunctionPolicy((s) -> rand(rng, actions(mdp, s))), Nsamples, Ntrials, rng)
zero_guess_residuals = bellman_residual(mdp, Bellman_states, (s)->0)
println("Importance Sampling Rollouts failed: ", is_failures, " ± ", is_std, " / ", Nsamples, " Bellman Residual: max=", maximum(zero_guess_residuals), " mean=", mean(zero_guess_residuals))

######### Subproblem Estimation Approaches (No global Approximation) ##########

# Utility Fusion with Mean
mean_uf_failures, mean_uf_std = run_trials(mdp, is_policy_subprob_mean, Nsamples, Ntrials, rng)
mean_uf_residuals = bellman_residual(mdp, Bellman_states, (s) -> value(is_policy_subprob_mean, s))
println("Mean Utility Fusion Rollouts failed: ", mean_uf_failures, " ± ", mean_uf_std, " / ", Nsamples,  " Bellman Residual: max=", maximum(mean_uf_residuals), " mean=", mean(mean_uf_residuals))


# Utility Fusion with Max
max_uf_failures, max_uf_std = run_trials(mdp, is_policy_subprob_max, Nsamples, Ntrials, rng)
max_uf_residuals = bellman_residual(mdp, Bellman_states, (s) -> value(is_policy_subprob_max, s))
println("Max Utility Fusion Rollouts failed: ", max_uf_failures, " ± ", max_uf_std, " / ", Nsamples,  " Bellman Residual: max=", maximum(max_uf_residuals), " mean=", mean(max_uf_residuals))


# Utility Fusion with Min
min_uf_failures, min_uf_std = run_trials(mdp, is_policy_subprob_min, Nsamples, Ntrials, rng)
min_uf_residuals = bellman_residual(mdp, Bellman_states, (s) -> value(is_policy_subprob_min, s))
println("Min Utility Fusion Rollouts failed: ", min_uf_failures, " ± ", min_uf_std, " / ", Nsamples,  " Bellman Residual: max=", maximum(min_uf_residuals), " mean=", mean(min_uf_residuals))


######### Train the global approximation ##########
iterations = 20
eps_per_iteration = 50

######### Subproblem Estimation Approaches (No global Approximation) ##########
# No Estimation -- Just global approximation trained with IS
println("Training Global Approx Model (No Estimate)... ")
mc_policy_eval!(is_policy_no_estimate, iterations, eps_per_iteration, verbose = false)
no_est_ga_failures, no_est_ga_std = run_trials(mdp, is_policy_no_estimate, Nsamples, Ntrials, rng)
no_est_ga_residuals = bellman_residual(mdp, Bellman_states, (s) -> value(is_policy_no_estimate, s))
println("No Estimate - Global Approximation Failed: ", no_est_ga_failures, " ± ", no_est_ga_std, " / ", Nsamples,  " Bellman Residual: max=", maximum(no_est_ga_residuals), " mean=", mean(no_est_ga_residuals))


# Utility Fusion with Mean + Global Approximation
println("Training Global Approx Model (No Mean UF)... ")
mc_policy_eval!(is_policy_subprob_mean, iterations, eps_per_iteration, verbose = false)
mean_uf_ga_failures, mean_uf_ga_std = run_trials(mdp, is_policy_subprob_mean, Nsamples, Ntrials, rng)
mean_uf_ga_residuals = bellman_residual(mdp, Bellman_states, (s) -> value(is_policy_subprob_mean, s))
println("Mean Utility Fusion Rollouts failed: ", mean_uf_ga_failures, " ± ", mean_uf_ga_std, " / ", Nsamples,  " Bellman Residual: max=", maximum(mean_uf_ga_residuals), " mean=", mean(mean_uf_ga_residuals))


# Utility Fusion with Max + Global Approximation
println("Training Global Approx Model (No Max UF)... ")
mc_policy_eval!(is_policy_subprob_max, iterations, eps_per_iteration, verbose = false)
max_uf_ga_failures, max_uf_ga_std = run_trials(mdp, is_policy_subprob_max, Nsamples, Ntrials, rng)
max_uf_ga_residuals = bellman_residual(mdp, Bellman_states, (s) -> value(is_policy_subprob_max, s))
println("Max Utility Fusion Rollouts failed: ", max_uf_ga_failures, " ± ", max_uf_ga_std, " / ", Nsamples,  " Bellman Residual: max=", maximum(max_uf_ga_residuals), " mean=", mean(max_uf_ga_residuals))


# Utility Fusion with Min + Global Approximation
println("Training Global Approx Model (No Min UF)... ")
mc_policy_eval!(is_policy_subprob_min, iterations, eps_per_iteration, verbose = false)
min_uf_ga_failures, min_uf_ga_std = run_trials(mdp, is_policy_subprob_min, Nsamples, Ntrials, rng)
min_uf_ga_residuals = bellman_residual(mdp, Bellman_states, (s) -> value(is_policy_subprob_min, s))
println("Min Utility Fusion Rollouts failed: ", min_uf_ga_failures, " ± ", min_uf_ga_std, " / ", Nsamples,  " Bellman Residual: max=", maximum(min_uf_ga_residuals), " mean=", mean(min_uf_ga_residuals))


##### Play some videos of the policies #######
h_mc = POMDPSimulators.simulate(HistoryRecorder(rng = rng), mdp, FunctionPolicy((s) -> random_action(mdp, s, rng)))
make_interact(state_hist(h_mc), mdp.models, mdp.roadway, egoid = mdp.egoid)

h_is = POMDPSimulators.simulate(HistoryRecorder(rng = rng), mdp, FunctionPolicy((s) -> rand(rng, actions(mdp, s))))
make_interact(state_hist(h_is), mdp.models, mdp.roadway, egoid = mdp.egoid)

h1 = POMDPSimulators.simulate(HistoryRecorder(rng = rng), decomposed[1], policies[1])
make_interact(state_hist(h1), decomposed[1].models, decomposed[1].roadway, egoid = decomposed[1].egoid)

h2 = POMDPSimulators.simulate(HistoryRecorder(rng = rng), decomposed[2], policies[2])
make_interact(state_hist(h2), decomposed[2].models, decomposed[2].roadway, egoid = decomposed[2].egoid)

h3 = POMDPSimulators.simulate(HistoryRecorder(rng = rng), decomposed[3], policies[3])
make_interact(state_hist(h3), decomposed[3].models, decomposed[3].roadway, egoid = decomposed[3].egoid)

h4 = POMDPSimulators.simulate(HistoryRecorder(rng = rng), decomposed[4], policies[4])
make_interact(state_hist(h4), decomposed[4].models, decomposed[4].roadway, egoid = decomposed[4].egoid)
