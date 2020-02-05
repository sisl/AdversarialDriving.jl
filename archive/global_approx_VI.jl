include("fitting.jl")
using POMDPSimulators
using Random
using POMDPModels
using POMDPs
using LocalFunctionApproximation
using GlobalApproximationValueIteration
using StaticArrays
using Printf
using Test
using POMDPModelTools
include("../solver/local_approx_Qp.jl")

# function sample_actions(mdp, s, N_actions, action_taken)
#     as = collect(actions(mdp, s))
#     N_actions = min(N_actions, length(as))
#     ret_as = as[randperm(length(as))[1:N_actions]]
#     if !any(ret_as .== action_taken)
#         ret_as[1] = action_taken
#     end
#     ret_as
# end

# function global_approx_VI(mdp, policy, model, max_iterations, N_actions, compute_value, Neps)
#     # Setup the x and y vectors for later fitting
#     xpts = Array{Float64, 2}(undef, N, nd)
#     ypts = Vector{Float64}(undef, N)
#
#     # Loop over the desired number of iterations
#     for iter in 1:max_iterations
#         # Get a history of states and actions
#         histories = [simulate(HistoryRecorder(max_steps = 1000), mdp, policy) for e in 1:Neps]
#         sh = vcat([state_hist(history)[1:end-1] for history in histories]...)
#         ah = vcat([collect(action_hist(history)) for history in histories]...)
#         Gh = vcat([return_hist(history) for history in histories]...)
#
#         N = length(sh)
#         nd = length(convert_s(mdp, sh[1]))
#
#         residual = 0
#         i = 1
#         # Loop through each state in the history (in reverse)
#         for si in N:-1:1
#             s, a = sh[si], ah[si]
#
#             # s = sample_state(g)
#
#             xpts[1, :] = convert_s(mdp, s)
#             ypts[1] = 0.
#
#             # Sample a bunch of actions, including the action that was taken
#             as = sample_actions(mdp, s, N_actions, a)
#             old_val = forward(model, convert_s(mdp, s)').data[1]
#
#             tot_p = 0
#
#             # Compute the probability of failure for the state and action
#             for a in as
#                 sp, r = gen(DDNOut(:sp,:r), mdp, s, a, solver.rng)
#                 u = r
#
#                 if !isterminal(mdp,sp)
#                     u += forward(model, convert_s(mdp, sp)').data[1]
#                 end
#                 p = action_probability(mdp, s, a)
#                 tot_p += p
#                 ypts[i] += u*p
#             end
#             ypts[i] /= tot_p
#             residual = abs(ypts[i] - old_val)
#             fit!(model, xpts, ypts)
#
#             # println("s: ", s, " old val: ", old_val, " new val: ", ypts[i])
#             if residual > max_residual
#                 max_residual = residual
#             end
#             # i += 1
#         end
#
#         println("i= ", iter, " residual: ", max_residual)
#         # fit!(model, xpts, ypts)
#     end
#     model
# end




#### Test it out ######
# Create the problem to be solved
g = SimpleGridWorld(size = (9,9), rewards = Dict(GWPos(9,9) => 1, GWPos(1,1) => 0), tprob = 1., discount=1)

# define the necessary functions
action_probability(g::SimpleGridWorld, s, a) = 0.25

function POMDPs.convert_s(g::SimpleGridWorld, s)
    x = s[1]
    y = s[2]

    v = SVector{4, Float64}(1, x, y, x*y)
end



POMDPs.gen(d::DDNOut{(:sp,:r)}, mdp::SimpleGridWorld, s, a, rng = Random.GLOBAL_RNG) = (sp =rand(transition(g, s, a )), r=reward(g, s, a))

struct RandPolicy <: Policy
end
POMDPs.action(p::RandPolicy, s::GWPos) = rand(actions(g))
POMDPs.value(policy::RandPolicy, s::GWPos, a::Symbol) = 0.25

# Solve it using the exact method (no function approximation)
grid = RectangleGrid([1:9 ...], [1:9 ...])

interp = LocalGIFunctionApproximator(grid)  # Create the local function approximator using the grid
solver = LocalQpSolver(interp, verbose = true, max_iterations = 2000, belres = 1e-6)
policy = solve(solver, g)
render(g, (s=GWPos(1,1), r=1), color = (s) -> 20. *(value(policy, s) - 0.5))

# Solve it with a linear function approximator
NN = init_MLP([4, 1])
model, X, y = mc_policy_eval(g, RandPolicy(), NN, 1, 100, 10)


N = length(y)
d = Dict{GWPos, Array{Float64}}()

for i=1:N
    pos = GWPos(X[i,2], X[i,3])
    if haskey(d, pos)
        push!(d[pos], y[i])
    else
        d[pos] = [y[i]]
    end
end

function get_val(d,s)
    if haskey(d,s)
        return mean(d[s])
    else
        return 0
    end
end


lin_model = LinearModel(4)
fit!(lin_model, X, y)
fit!(NN, X, y, 10)

sum((forward(NN, X) .- y).^2)


render(g, (s=GWPos(1,1), r=1), color = (s) -> 20. *(get_val(d,s)- 0.5))
render(g, (s=GWPos(1,1), r=1), color = (s) -> 20. *(forward(NN, convert_s(g, s)').data[1]- 0.5))
render(g, (s=GWPos(1,1), r=1), color = (s) -> 20. *(forward(lin_model, convert_s(g, s)')- 0.5))

NN(convert_s(g, GWPos(9,9)))
