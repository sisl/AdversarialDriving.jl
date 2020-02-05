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
POMDPs.convert_s(g::SimpleGridWorld, s) = SVector{4, Float64}(1., s[1], s[2], s[1]*s[2])
POMDPs.gen(d::DDNOut{(:sp,:r)}, mdp::SimpleGridWorld, s, a, rng = Random.GLOBAL_RNG) = (sp =rand(transition(g, s, a )), r=reward(g, s, a))


# Step 2 - Solve the problem semi-exactly using local approximation
grid = RectangleGrid([1:g_size[1] ...], [1:g_size[2]...])
interp = LocalGIFunctionApproximator(grid)
solver = LocalPolicyEvalSolver(interp, verbose = true, max_iterations = 2000, belres = 1e-6)
policy = solve(solver, g)
values = [value(policy, s) for s in states(g)]
p1 = render(g, (s=GWPos(1,1), r=1), color = (s) -> 20. *(value(policy, s) - 0.5))
draw(PDF("Gridworld_groundtruth.pdf"), p1)


# Step 3 - Solve the problem using the Linear estimator -> progressively incrementing it?\
#  Get the ideal linear paramters
X = to_mat([convert_s(g, s) for s in states(g)])
ideal_model = LinearModel(4)
fit!(ideal_model, X, values)
ideal_model_values = [forward(ideal_model, convert_s(g, s)') for s in states(g)]
ideal_mse = sum((ideal_model_values .- values).^2) / length(values)

mutable struct ISPolicy <: Policy
    mdp
    model
end

update_policy!(p::ISPolicy, model) = p.model = model

function POMDPs.action(p::ISPolicy, s::GWPos, rng = Random.GLOBAL_RNG)
    as = actions(p.mdp, s)
    Na = length(as)
    pf = Array{Float64}(undef, Na)
    for i=1:Na
        a = as[i]
        sp, r = gen(DDNOut((:sp,:r)), p.mdp, s, a, rng)
        pf[i] = action_probability(p.mdp, s, a)*forward(model, convert_s(p.mdp, sp)')
        pf[i] = min(max(0, pf[i]), 1)
    end
    sum_pf = sum(pf)
    pf = (sum_pf == 0) ? ones(Na)/Na : pf/sum_pf
    ai = rand(Categorical(pf))
    as[ai], pf[ai]
end


s_rand = convert_s(g,rand(initialstate_distribution(g)))
model = LinearModel(length( s_rand))
is_policy = ISPolicy(g, model)

errs = []
for i=1:100
    is_values = [forward(model, convert_s(g, s)') for s in states(g)]
    push!(errs, sum((is_values .- values).^2) / length(values))
    mc_policy_eval(g, is_policy, model, 1, 1)
end

p2 = render(g, (s=GWPos(1,1), r=1), color = (s) -> 20. *(forward(model, convert_s(g, s)')- 0.5))
draw(PDF("Gridworld_linear_approx.pdf"), p2)

plot(errs, label = "Online linear model", ylabel="MSE", xlabel="Iteration", title="Convergance of Linear Policy Eval")
plot!(ones(length(errs))*ideal_mse, label="Ideal MSE")

