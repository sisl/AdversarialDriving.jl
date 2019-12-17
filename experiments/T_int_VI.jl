include("plot_utils.jl")
include("../simulator/adm_task_generator.jl")
using POMDPs
using GridInterpolations
using LocalFunctionApproximation
using LocalApproximationValueIteration

task = generate_1car_ADM_POMDP(dt = 0.3)
get_by_id(task.initial_scene,1).state.veh_state.v

N = 15
grid = RectangleGrid(
# Vehicle 1
    range(35, stop=75,length=N), # position
    range(15, stop=30., length=N), # Velocity
    [2.0], # Goal
    [0.0, 1.0], # Blinker
# Vehicle 2
    range(25, stop=75, length=N), # position
    range(0., stop=25., length=N), # Velocity
    [5.0], # Goal
    [0.0], # Blinker
    )

interp = LocalGIFunctionApproximator(grid)  # Create the local function approximator using the grid

solver = LocalApproximationValueIterationSolver(interp, is_mdp_generative = true, n_generative_samples = 1, verbose = true, max_iterations = 20)

policy = solve(solver, task)

using Serialization
serialize("policy_works", policy)

a = value(policy, task.initial_scene)

o,a,r,scenes = policy_rollout(task, (o) -> action(policy, convert_s(BlinkerScene, o, task)), task.initial_scene, save_scenes = true)

action_to_string.(Int.(a))

make_interact(scenes, task.models, task.roadway; egoid = task.egoid)

