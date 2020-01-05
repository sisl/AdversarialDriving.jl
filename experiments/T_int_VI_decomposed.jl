include("../simulator/adm_task_generator.jl")
using POMDPs
using GridInterpolations
using LocalFunctionApproximation
include("../solver/local_approx_Qp.jl")
using Serialization

decomposed, combined = generate_decomposed_scene(dt = 0.18)
policies = Array{Any}(undef, length(decomposed))
N = 20

for i in 1:length(decomposed)
    println("Solving decomposition ", i)
    pomdp = decomposed[i]
    veh = get_by_id(pomdp.initial_scene, 1)

    grid = RectangleGrid(
    # Vehicle 1
        range(posf(veh.state).s, stop=63,length=N), # position
        range(0, stop=29., length=N), # Velocity
        pomdp.models[1].goals[laneid(veh)], # Goal
        [0.0, 1.0], # Blinker
    # Vehicle 2
        range(35, stop=65, length=N), # position
        range(0., stop=25., length=N), # Velocity
        [5.0], # Goal
        [1.0], # Blinker
        )

    interp = LocalGIFunctionApproximator(grid)  # Create the local function approximator using the grid

    solver = LocalQpSolver(interp, is_mdp_generative = true, n_generative_samples = 1, verbose = true, max_iterations = 100, belres = 1e-6)

    policy = solve(solver, pomdp)
    serialize(string("policy_decomp_", i, ".jls"), policy)
    policies[i] = policy
end

serialize("combined_policies.jls", policies)