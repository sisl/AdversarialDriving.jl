include("../simulator/adm_task_generator.jl")
include("../solver/linear_mc_policy_eval.jl")
include("../solver/local_approx_policy_eval.jl")
using Serialization
using Plots

decomposed, combined = generate_decomposed_scene(dt = 0.18)
policies = [deserialize(string("policy_decomp_", i, ".jls")) for i in 1:length(decomposed)]

# Define the subproblem estimates with different approaches to utility fusion
Vest_mean = subproblem_estimate_fn(policies, (s) -> decompose_scene(s, combined.egoid), :mean)
Vest_min = subproblem_estimate_fn(policies, (s) -> decompose_scene(s, combined.egoid), :min)
Vest_max = subproblem_estimate_fn(policies, (s) -> decompose_scene(s, combined.egoid), :max)

Ns = length(convert_s(Vector{Float64},  initialstate(combined), combined))
is_policy_no_estimate = ISPolicy(combined, LinearModel(Ns), (s) -> 0)
is_policy_subprob_mean = ISPolicy(combined, LinearModel(Ns), Vest_mean)
is_policy_subprob_min = ISPolicy(combined, LinearModel(Ns), Vest_min)
is_policy_subprob_max = ISPolicy(combined, LinearModel(Ns), Vest_max)

N_iter = 15
N_eps_per_it = 100
p = plot(title="Failure Rate of Policy", xlabel = string("Number of iterations (", N_eps_per_it, " eps each)"), ylabel="Failure Fraction")
colors = [:red, :blue, :black, :green]
policies_to_test = [is_policy_no_estimate, is_policy_subprob_mean, is_policy_subprob_min, is_policy_subprob_max]
policy_names = ["No estimate", "Estimate w/ Subproblem Mean", "Estimate w/ Subproblem Min", "Estimate w/ Subproblem Max"]
N_pol = length(policies_to_test)
failure_rates = []
for i=1:N_pol
    println("Evaluating policy: ", policy_names[i])
    pol_to_evaluate = policies_to_test[i]
    failure_rate = []
    for i=1:N_iter
        println("    Evaluating policy after iteration ", i-1)
        mc_policy_eval!(pol_to_evaluate, 1, N_eps_per_it, verbose = false, failure_rate_vec = failure_rate)
    end
    push!(failure_rates, failure_rate)
    plot!(p, 1:N_iter, failure_rate, label = policy_names[i], linecolor = colors[i])
end
plot(p)

savefig("T_intersection_comparisons.pdf")

