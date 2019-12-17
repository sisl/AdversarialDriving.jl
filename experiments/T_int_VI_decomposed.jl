include("plot_utils.jl")
include("../simulator/adm_task_generator.jl")
using POMDPs
using GridInterpolations
using LocalFunctionApproximation
using LocalApproximationValueIteration

decomposed, combined = generate_decomposed_scene(dt = 0.18)
goals = combined.models[1].goals
policies = Array{Any}(undef, length(decomposed))
N = 3


for i in 1:length(decomposed)
    println("Solving decomposition ", i)
    pomdpd = decomposed[i]
    veh = get_by_id(pomdp.initial_scene, 1)

    grid = RectangleGrid(
    # Vehicle 1
        range(posf(veh.state).s, stop=75,length=N), # position
        range(0, stop=29., length=N), # Velocity
        goals[laneid(non_ego_vehicle)], # Goal
        [0.0, 1.0], # Blinker
    # Vehicle 2
        range(25, stop=75, length=N), # position
        range(0., stop=25., length=N), # Velocity
        [5.0], # Goal
        [1.0], # Blinker
        )

    interp = LocalGIFunctionApproximator(grid)  # Create the local function approximator using the grid

    solver = LocalApproximationValueIterationSolver(interp, is_mdp_generative = true, n_generative_samples = 1, verbose = true, max_iterations = 20)

    policy = solve(solver, pomdp)
    policies[i] = policy
end

policies

a = value(policy, task.initial_scene)

o,a,r,scenes = policy_rollout(task, (o) -> action(policy, convert_s(BlinkerScene, o, task)), task.initial_scene, save_scenes = true)

action_to_string.(Int.(a))

make_interact(scenes, task.models, task.roadway; egoid = task.egoid)

