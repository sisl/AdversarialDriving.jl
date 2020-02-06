include("../solver/linear_mc_policy_eval.jl")
include("../solver/local_approx_policy_eval.jl")
using POMDPModels
using POMDPModelTools
using StaticArrays
using Profile
using Compose
using Cairo, Fontconfig
using Plots


# Step 1 - Setup the gridworld problem
g_size = (9,9)
g = SimpleGridWorld(size = g_size, rewards = Dict(GWPos(g_size...) => 1, GWPos(1,1) => 0), tprob = 1., discount=1)
action_probability(g::SimpleGridWorld, s, a) = 0.25
POMDPs.convert_s(::Type{AbstractArray}, s::GWPos, g::SimpleGridWorld) = SVector{4, Float64}(1., s[1], s[2], s[1]*s[2])
POMDPs.gen(d::DDNOut{(:sp,:r)}, mdp::SimpleGridWorld, s, a, rng = Random.GLOBAL_RNG) = (sp =rand(transition(g, s, a )), r=reward(g, s, a))
POMDPs.initialstate(g::SimpleGridWorld) = rand(initialstate_distribution(g))


# Step 2 - Solve the problem semi-exactly using local approximation
grid = RectangleGrid([1:g_size[1] ...], [1:g_size[2]...])
interp = LocalGIFunctionApproximator(grid)
solver = LocalPolicyEvalSolver(interp, is_mdp_generative = true, n_generative_samples = 1,  verbose = true, max_iterations = 2000, belres = 1e-6)
policy = solve(solver, g)
values = [value(policy, s) for s in states(g)]

p1 = render(g, (s=GWPos(1,1), r=1), color = (s) -> 20. *(value(policy, s) - 0.5))
draw(PDF("Gridworld_groundtruth.pdf"), p1)


# Step 3 - Solve the problem using the Linear estimator -> progressively incrementing it?\
#  Get the ideal linear paramters
Xall = to_mat([convert_s(AbstractArray, s, g) for s in states(g)])
ideal_model = LinearModel(4)
fit!(ideal_model, Xall, values)
ideal_model_values = [forward(ideal_model, convert_s(AbstractArray, s, g)') for s in states(g)]
ideal_mse = mse(ideal_model_values, values)


s_rand = convert_s(AbstractArray, rand(initialstate_distribution(g)), g)
model = LinearModel(length(s_rand))
rand_est = Dict(s => rand() for s in states(g))
is_policy = ISPolicy(g, model, (mdp, s) -> 0.5)

errs = []
for i=1:10
    is_values = [value(is_policy, s) for s in states(g)]
    push!(errs, mse(is_values, values))
    mc_policy_eval(g, is_policy, 1, 10)
end

push!(errs, mse(is_values, values))

p2 = render(g, (s=GWPos(1,1), r=1), color = (s) -> 20. *(value(is_policy, s)- 0.5))
draw(PDF("Gridworld_linear_approx.pdf"), p2)

plot(errs, label = "Online linear model", ylabel="MSE", xlabel="Iteration", title="Convergance of Linear Policy Eval")
plot!(ones(length(errs))*ideal_mse, label="Ideal MSE")
savefig("Gridworld_convergence_of_linear_model.pdf")

