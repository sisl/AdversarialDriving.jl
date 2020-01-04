using POMDPs
using POMDPModels
using POMDPModelTools
using DiscreteValueIteration
using Random


gridworld(i,j; size = (10,10)) = SimpleGridWorld(size = size, rewards = Dict(GWPos(i,j) => 1), tprob = 1.)
function POMDPs.gen(pomdp::SimpleGridWorld, s, a, rng = Random.GLOBAL_RNG)
    res = transition(pomdp, s, a)
    sp = (res isa Deterministic) ? res.val : res.vals[rand(rng, Categorical(res.probs))]
    r = reward(g,s,a,sp)
    sp, r
end

S, A, Sp, R  = sample_all_states(g)


g = gridworld(10,10)

solver = ValueIterationSolver(max_iterations=100, belres=1e-6, verbose = true) # creates the solver
policy = solve(solver, g) # runs value iterations
value(policy, GWPos(10,10))

function tocolor(r::Float64)
    minr = 0
    maxr = 1.0
    frac = (r-minr)/(maxr-minr)
    return get(ColorSchemes.redgreensplit, frac)
end
render(g, (s=[7,6],), color = (s) -> 10*value(policy, s))

