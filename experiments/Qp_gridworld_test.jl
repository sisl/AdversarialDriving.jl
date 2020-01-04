using POMDPs
using POMDPModels
using POMDPModelTools
using POMDPPolicies
using Random
using Distributions
include("../solver/local_approx_Qp.jl")
using LocalApproximationValueIteration

# Construct a gridworld
g = SimpleGridWorld(size = (10,10), rewards = Dict(
    GWPos(5,5) => 1,
    GWPos(6,5) => 0,
    GWPos(7,5) => 0,
    GWPos(8,5) => 0,
    GWPos(9,5) => 0,
    GWPos(4,5) => 0,
    GWPos(5,6) => 0,
    GWPos(5,3) => 0,
    ), tprob = 1., discount=1.)
s0 = [3,3]

action_probability(mdp::SimpleGridWorld, a) = 0.25

# Plot the gridworld
render(g, (s=s0,), color = (s) -> 20*reward(g,s) -10. *(s in g.terminate_from))

# Solve for Q function
grid = RectangleGrid(
    [1:10 ...],
    [1:10 ...]
    )

interp = LocalGIFunctionApproximator(grid)  # Create the local function approximator using the grid

solver = LocalQpSolver(interp, is_mdp_generative = true, n_generative_samples = 1, verbose = true, max_iterations = 1000, belres = 1e-6)

policy = solve(solver, g)

render(g, (s=s0,), color = (s) -> 10*value(policy, s))
value(policy, s0)

# Perform MC sampling to get an estimate of the probability of success
N = 100000
num_success = 0
for t in 1:N
    println("trial: ", t)
    # Perform the rollout
    s = s0
    while !isterminal(g,s)
        a = rand(actions(g, s))
        sp = rand(transition(g,s,a))
        if reward(g, s, a, sp) == 1
            global num_success += 1
        end
        s = sp
    end
end
succ_prob = num_success / N

# Create policy that samples according to Qp
s = s0
state_arr = []
last_r = NaN
while !isterminal(g, s)
    push!(state_arr, s)
    a = action(policy, s, true)
    sp = rand(transition(g,s,a))
    r = reward(g, s, a, sp)
    if r != 0
        println("reward: ", r)
    end
    global s = sp
end
state_arr
s

# Check fraction of success (should be 100)

# Compute probability of failure via sampling