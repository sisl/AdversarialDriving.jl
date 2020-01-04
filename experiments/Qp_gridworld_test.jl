using POMDPs
using POMDPModels
using POMDPModelTools
using POMDPPolicies
using DiscreteValueIteration
using Random

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
s0 = [6,6]

s0 in g.terminate_from
# Plot the gridworld
render(g, (s=s0,), color = (s) -> 20*reward(g,s) -10. *(s in g.terminate_from))

# Solve for Q function
solver = ValueIterationSolver(max_iterations=100, belres=1e-6, verbose = true) # creates the solver
policy = solve(solver, g) # runs value iterations
value(policy, GWPos(10,10))
render(g, (s=s0,), color = (s) -> 10*value(policy, s))

# Create policy that samples according to Qp
s = s0
state_arr = []
last_r = NaN
while !isterminal(g, s)
    push!(state_arr, s)
    Qs = actionvalues(policy, s)
    a = actions(g)[rand(Categorical(Qs / sum(Qs)))]
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