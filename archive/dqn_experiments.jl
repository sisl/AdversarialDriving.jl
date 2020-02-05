using POMDPModels
using POMDPs
using DeepQLearning
using Flux
using LocalFunctionApproximation
using POMDPSimulators
include("../solver/local_approx_Qp.jl")
# define the necessary functions
action_probability(g::SimpleGridWorld, s, a) = 0.25
POMDPs.convert_s(g::SimpleGridWorld, s) = SVector{4, Float64}(1, s[1], s[2], s[1]*s[2])
POMDPs.gen(d::DDNOut{(:sp,:r)}, mdp::SimpleGridWorld, s, a, rng = Random.GLOBAL_RNG) = (sp =rand(transition(g, s, a )), r=reward(g, s, a))


g = SimpleGridWorld(size = (9,9), rewards = Dict(GWPos(9,9) => 1,GWPos(1,1) => 0,), tprob = 1., discount=0.8)

# Solve it using local function approximation
interp = LocalGIFunctionApproximator(RectangleGrid([1:9 ...], [1:9 ...]))
solver = LocalQpSolver(interp, verbose = true, max_iterations = 2000, belres = 1e-6)
policy = solve(solver, g)

render(g, (s=GWPos(1,1), r=1), color = (s) -> 20. *(value(policy, s) - 0.5))

# Solve it using DQN
model = Chain(Dense(2, length(actions(g))))

solver = DeepQLearningSolver(qnetwork = model, max_steps=10000,
                             learning_rate=0.01,log_freq=500,
                             recurrence=false, double_q=false, dueling=false, prioritized_replay=true)

policy = solve(solver, g)

getnetwork(policy)(GWPos(8,9))
value(policy, GWPos(9,9))
value(policy, GWPos(1,1))

render(g, (s=GWPos(1,1), r=1), color = (s) -> 20. *(value(policy, s).data - 0.5))


sim = RolloutSimulator(max_steps=30)
r_tot = simulate(sim, g, policy)
println("Total discounted reward for 1 simulation: $r_tot")

