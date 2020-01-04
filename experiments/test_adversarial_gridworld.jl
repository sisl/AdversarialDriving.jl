include("../simulator/adversarial_gridworld.jl")
using Plots
using DiscreteValueIteration

mdp, V, g_fail = create_sim(AdversarialGridWorld, w=10, h=10)

_, π0 = value_iteration(g_fail)
display_gridworld(mdp.g, policy_to_annotations(π0))

solver = ValueIterationSolver(max_iterations=100, belres=1e-6, verbose = true) # creates the solver
policy = solve(solver, mdp) # runs value iterations
value(policy, GWPos(10,10))

