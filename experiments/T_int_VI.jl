include("plot_utils.jl")
include("../simulator/adm_task_generator.jl")
using POMDPs
using GridInterpolations
using LocalFunctionApproximation
using LocalApproximationValueIteration

task = generate_1car_ADM_POMDP(dt = 0.18)

N = 5
grid = RectangleGrid(
# Vehicle 1
    range(25, stop=55,length=N), # position
    range(15, stop=30., length=N), # Velocity
    [1.0, 2.0], # Goal
    [0.0, 1.0], # Blinker
# Vehicle 2
    range(35, stop=55, length=N), # position
    range(0., stop=10., length=N), # Velocity
    [5.0], # Goal
    [0.0, 1.0], # Blinker
    )

interp = LocalGIFunctionApproximator(grid)  # Create the local function approximator using the grid

solver = LocalApproximationValueIterationSolver(interp, is_mdp_generative = true, n_generative_samples = 1, verbose = true)

policy = solve(solver, task)

a = value(policy, task.initial_scene) # returns the approximately optimal action for state s


o,a,r,scenes = policy_rollout(task, (o) -> action(policy, convert_s(BlinkerScene, o, task)), task.initial_scene, save_scenes = true)

action_to_string.(Int.(a))

make_interact(scenes, task.models, task.roadway; egoid = task.egoid)

