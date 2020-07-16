using AdversarialDriving
using POMDPs, POMDPPolicies, POMDPSimulators
using Distributions
using GridInterpolations, LocalFunctionApproximation, LocalApproximationValueIteration
using Serialization

save_folder ="examples/training_hist/"
Np, Nv = 3, 3
N_suts = 10

# for i=1:N_suts
i = 1
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

interp = LocalGIFunctionApproximator(grid)  # Create the local function approximator using the grid
solver = LocalApproximationValueIterationSolver(interp, is_mdp_generative = true, n_generative_samples = 10, verbose = true, max_iterations = 25, belres = 1e-6)
policy = solve(solver, mdp)

folder = string(save_folder, "sut_version_", lpad(i, 2, "0"), "/")
try mkdir(folder) catch end
println("saving...")
serialize(string(folder, "mdp"), mdp)
serialize(string(folder, "policy"), policy)

println("Evaluating the performance...")

h = simulate(HistoryRecorder(max_steps = 150), mdp, policy)
scenes_to_gif(state_hist(h), mdp.roadway, "out.gif")
## Evaluate the performance
tot_suc, Ntrials =0, 10
for k=1:Ntrials
    println("k:", k)
    h = simulate(HistoryRecorder(max_steps = 150), mdp, policy)
    global tot_suc += any(reward_hist(h) .> 1.)
end
tot_suc
prob_suc =  tot_suc / Ntrials
println("success prob: ", prob_suc)
write("performance.txt", string(prob_suc))
# end
