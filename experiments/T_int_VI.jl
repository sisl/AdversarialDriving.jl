include("../simulator/adm_task_generator.jl")
using POMDPs
using GridInterpolations
using LocalFunctionApproximation
using LocalApproximationValueIteration

task = generate_1car_ADM_POMDP(dt = 0.18)

grid = RectangleGrid(
# Vehicle 1
    range(25, stop=75,length=10), # position
    range(0, stop=30., length=10), # Velocity
    [1.0, 2.0], # Goal
    [0.0, 1.0], # Blinker
# Vehicle 2
    range(15,stop=75,length=10), # position
    range(0., stop=30., length=10), # Velocity
    [5.0], # Goal
    [0.0, 1.0], # Blinker
    )

interp = LocalGIFunctionApproximator(grid)  # Create the local function approximator using the grid

solver = LocalApproximationValueIterationSolver(interp, is_mdp_generative = true, n_generative_samples = 1, verbose = true)

policy = solve(solver, task)

