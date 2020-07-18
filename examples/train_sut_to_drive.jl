using AdversarialDriving
using POMDPs, POMDPPolicies, POMDPSimulators
using Distributions
using GridInterpolations, LocalFunctionApproximation, LocalApproximationValueIteration

## setup training params
Np, Nv = 3,3 #25, 15
Nsteps = 2#50

## Construct a disturbance model for the adversaries
da_std = rand(Exponential(0.5))
goal_toggle_p = 10. ^rand(Uniform(-5., -1.))
blinker_toggle_p = 10. ^rand(Uniform(-5., -1.))
v_des = rand(Uniform(15., 30.))
per_timestep_penalty = rand([0, 1e-4, 1e-3, 5e-3])
disturbances = Sampleable[Normal(0,da_std), Bernoulli(goal_toggle_p), Bernoulli(blinker_toggle_p), Bernoulli(0.), Bernoulli(0.)]

## construct the MDP
sut_agent = BlinkerVehicleAgent(rand_up_left(id=1, s_dist=Uniform(15.,25.), v_dist=Uniform(10., 29)), TIDM(Tint_TIDM_template))
right_adv = BlinkerVehicleAgent(rand_right(id=3, s_dist=Uniform(15.,35.), v_dist=Uniform(15., 29.)), TIDM(Tint_TIDM_template), disturbance_model = disturbances)
mdp = DrivingMDP(sut_agent, [right_adv], Tint_roadway, 0.2, γ = 0.95, per_timestep_penalty = per_timestep_penalty, v_des = v_des)

## Solve using local approximation value iteration
goals = Float64.(Tint_goals[laneid(adversaries(mdp)[1].get_initial_entity())])
grid = RectangleGrid(range(0., stop=100.,length=Np), range(0, stop=30., length=Nv), goals, [0.0, 1.0],
                      range(15., stop=100., length=Np), range(0., stop=30., length=Nv), [5.0], [1.0])

interp = LocalGIFunctionApproximator(grid)
solver = LocalApproximationValueIterationSolver(interp, is_mdp_generative = true, n_generative_samples = 5, verbose = true, max_iterations = Nsteps)
policy = solve(solver, mdp)
