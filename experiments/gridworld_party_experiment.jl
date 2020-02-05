include("../simulator/gridworld_party.jl")
include("../solver/local_approx_Qp.jl")
using LocalApproximationValueIteration
using GridInterpolations
using LocalFunctionApproximation
using Profile
using POMDPSimulators
using Interact

struct VecPolicy1 <: Policy
    vec::Vector{LocalQpPolicy}
end

function solve_mdp(mdp, grid, n_generative_samples, max_iterations, is_probability)
    interp = LocalGIFunctionApproximator(grid)
    solver_t = is_probability ? LocalQpSolver : LocalApproximationValueIterationSolver
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
policy = solve_mdp(mdp, grid, 10, 20, false)

history = simulate(HistoryRecorder(max_steps = 100), mdp, policy)
@manipulate for t in 1:length(history)
    (s,a,r,sp) = history[t]
    render(mdp, s)
end

############# Step 2 - Solve for the adversarial policy of the gridworld ############
a_mdp.a_dict = action_dict(mdp, policy)
a_policy = solve_mdp(a_mdp, grid, 1, 100, true)

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
a_policies = Array{LocalQpPolicy}(undef, N)

# Solve for the 2D policies
for i in 1:N
    policies[i] = solve_mdp(decomps[i], grid2, 10, 20, false)
    a_decomps[i].a_dict = action_dict(decomps[i], policies[i])
    a_policies[i] = solve_mdp(a_decomps[i], grid2, 1, 100, true)
end

# Demonstrate the policy and adversarial policy of the subproblem in saction
si = 3

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



#Construct a policy and do rollouts for the combined problems
#NOTE: This implementation overights the actions of earlier pairings in favor of later pairings. This is lazy
function action(pol::VecPolicy1, s::Vector{GWPos})
    n_agents = length(s)
    indices = decompose_indices(n_agents)
    a = Vector{Symbol}(undef, n_agents)
    pf = zeros(n_agents)
    for i in 1:length(pol.vec)
        proposed_action = action(pol.vec[i], s[indices[i]])
        a[indices[i]] .= proposed_action
    end
    a
end

recomb_history = simulate(HistoryRecorder(max_steps = 100), a_mdp, VecPolicy1(a_policies))
@manipulate for t in 1:length(recomb_history)
    (s,a,r,sp) = recomb_history[t]
    render(a_mdp, s)
end

############ Step 4 - Perform a global correction #####################



function probability_failure(s::Vector{GWPos}, a::Vector{Symbol}, θ, pol)
    # Get the probability of failure estimate from the subproblems
    indices = decompose_indices(length(s))
    N_subproblems = length(indices)
    V = 0
    for i in 1:N_subproblems
        V += value(pol.vec[i], s[indices[i]], a[indices[i]])
    end
    V = V / N_subproblems


    # Add to it the estimate from the logistic regression model
    δV = 1. / (1. + exp(-sum(θ .* s)))

    V + δV
end

global_approx_VI(a_mdp, VecPolicy1(a_policies))







