include("../simulator/gridworld_party.jl")
include("../solver/local_approx_policy_eval.jl")
include("../solver/linear_mc_policy_eval.jl")
using LocalApproximationValueIteration
using GridInterpolations
using LocalFunctionApproximation
using Profile
using POMDPSimulators
using Interact
using Plots

# Functoin for solving an mdp with local function approximation value iteration
function solve_mdp(mdp, grid, n_generative_samples, max_iterations, is_probability)
    interp = LocalGIFunctionApproximator(grid)
    solver_t = is_probability ? LocalPolicyEvalSolver : LocalApproximationValueIterationSolver
    solver = solver_t(interp, is_mdp_generative = true, n_generative_samples = n_generative_samples, verbose = true, max_iterations = max_iterations, belres = 1e-6)
    solve(solver, mdp)
end

############ Setup - Construct the gridworld and the discretized statespace ####
n_agents = 3
Nx, Ny = 4,4
goals = [(1,1), (Nx,Ny), (1,Ny)][1:n_agents]
mdp = GridworldParty(size = (Nx, Ny), n_agents = n_agents, goals = goals, reward_type = :normal, tprob = 0.7, discount = 0.9) # Regular MDP
a_mdp = AdversarialGridworldParty(mdp)

grid = RectangleGrid( [1:(Nx, Ny)[mod1(i, 2)] for i in 1:2*n_agents]... ) # Full grid
grid2 = RectangleGrid( [1:(Nx, Ny)[mod1(i, 2)] for i in 1:4]... ) # Grid used for subproblem (2 agents)


############# Step 1 - Solve for the optimal policy  of the gridworld ############
policy = solve_mdp(mdp, grid, 10, 30, false)

h1 = simulate(HistoryRecorder(max_steps = 100), mdp, policy)
@manipulate for t in 1:length(h1)
    (s,a,r,sp) = h1[t]
    render(mdp, s)
end

############# Step 2 - Solve for the adversarial policy of the gridworld ############
# create a lookup for what actions the forward mdp agents would have taken
a_mdp.a_dict = action_dict(mdp, policy)
a_policy = solve_mdp(a_mdp, grid, 1, 50, true)

# Generate the ground truth probability of failure
pf_ground_truth_dict = Dict(s => value(a_policy, s) for s in states(a_mdp))
pf_ground_truth = [value(a_policy, s) for s in states(a_mdp)]

a_history = simulate(HistoryRecorder(max_steps = 100), a_mdp, a_policy)
@manipulate for t in 1:length(a_history)
    (s,a,r,sp) = a_history[t]
    render(a_mdp, s)
end

############ Step 3 - Decompose and solve subproblems #######################
decomps = decompose(mdp)
a_decomps = decompose(a_mdp)
N = length(decomps)

policies = Array{LocalApproximationValueIterationPolicy}(undef, N)
a_policies = Array{LocalPolicyEvalPolicy}(undef, N)

# Solve for the 2D policies
for i in 1:N
    policies[i] = solve_mdp(decomps[i], grid2, 10, 20, false)
    a_decomps[i].a_dict = action_dict(decomps[i], policies[i])
    a_policies[i] = solve_mdp(a_decomps[i], grid2, 1, 100, true)
end

# Demonstrate the policy and adversarial policy of the subproblem in saction
si = 1

s_history = simulate(HistoryRecorder(max_steps = 100), decomps[si], policies[si])
@manipulate for t in 1:length(s_history)
    (s,a,r,sp) = s_history[t]
    render(decomps[si], s)
end

a_s_history = simulate(HistoryRecorder(max_steps = 100), a_decomps[si], a_policies[si])
@manipulate for t in 1:length(a_s_history)
    (s,a,r,sp) = a_s_history[t]
    render(a_decomps[si], s)
end


############ Step 4 - Perform a global correction #####################
Vest_mean = subproblem_estimate_fn(policies, decompose_state, :mean)
Vest_min = subproblem_estimate_fn(policies, decompose_state, :min)
Vest_max = subproblem_estimate_fn(policies, decompose_state, :max)

subproblem_values_mean = [Vest_mean(s) for s in states(a_mdp)]
err_subproblem_mean = mse(subproblem_values_mean, pf_ground_truth)
subproblem_values_min = [Vest_min(s) for s in states(a_mdp)]
err_subproblem_min = mse(subproblem_values_mean, pf_ground_truth)
subproblem_values_max = [Vest_max(s) for s in states(a_mdp)]
err_subproblem_max = mse(subproblem_values_max, pf_ground_truth)


s_rand = convert_s(Vector{Float64},  initialstate(a_mdp), a_mdp)
is_policy_no_estimate = ISPolicy(a_mdp, LinearModel(length(s_rand)), (s) -> 0)
is_policy_subprob_mean = ISPolicy(a_mdp, LinearModel(length(s_rand)), Vest_mean)
is_policy_subprob_min = ISPolicy(a_mdp, LinearModel(length(s_rand)), Vest_min)
is_policy_subprob_max = ISPolicy(a_mdp, LinearModel(length(s_rand)), Vest_max)

N_iter = 15
N_eps_per_it = 100
p1 = plot(title="Convergence of Probability Models", xlabel = string("Number of iterations (", N_eps_per_it, " eps each)"), ylabel="MSE")
p2 = plot(title="Failure Rate of Policy", xlabel = string("Number of iterations (", N_eps_per_it, " eps each)"), ylabel="Failure Fraction")
X = to_mat([convert_s(AbstractArray, s, a_mdp) for s in states(a_mdp)])
colors = [:red, :blue, :black, :green]
policies_to_test = [is_policy_no_estimate, is_policy_subprob_mean, is_policy_subprob_min, is_policy_subprob_max]
policy_names = ["No estimate", "Estimate w/ Subproblem Mean", "Estimate w/ Subproblem Min", "Estimate w/ Subproblem Max"]
N_pol = length(policies_to_test)
errs = []
for i=1:N_pol
    println("Evaluating policy: ", policy_names[i])
    pol_to_evaluate = policies_to_test[i]
    err = []
    failure_rate = []
    for i=1:N_iter
        println("    Evaluating policy after iteration ", i-1)
        is_values = [value(pol_to_evaluate, s) for s in states(a_mdp)]
        push!(err, mse(is_values, pf_ground_truth))
        mc_policy_eval!(pol_to_evaluate, 1, N_eps_per_it, verbose = false, failure_rate_vec = failure_rate)
    end
    println("    Evaluating policy after iteration ", N_iter)
    is_values = [value(pol_to_evaluate, s) for s in states(a_mdp)]
    push!(err, mse(is_values, pf_ground_truth))
    push!(errs, err)
    plot!(p1, 0:N_iter, err, label = policy_names[i], linecolor = colors[i])
    plot!(p2, 1:N_iter, failure_rate, label = policy_names[i], linecolor = colors[i])

    # Determine the best that this policy could have done
    # ideal_model = LinearModel(length(s_rand))
    # estimates = [pol_to_evaluate.estimate(a_mdp, s) for s in states(a_mdp)]
    # fit!(ideal_model, X, pf_ground_truth .- estimates)
    # ideal_model_vals = max.(0, min.(1, estimates .+ [forward(ideal_model, convert_s(AbstractArray, s, a_mdp)') for s in states(a_mdp)]))
    # ideal_err = mse(ideal_model_vals, pf_ground_truth)
    # plot!(0:N_iter, ones(N_iter+1)*ideal_err, label = string(policy_names[i], " -- ideal"), linecolor = colors[i], linestyle = :dash)
end
plot(plot(p1, yscale = :log), plot(p2), size = (1200, 400))

errs

savefig("gridworld_party_comparisons.pdf")

