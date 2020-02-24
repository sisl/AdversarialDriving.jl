using ExprOptimization
using ExprRules
include("../../ltl_sampling/LTLSampling.jl")
include("../simulator/adm_task_generator.jl")
using Plots; gr()

# Define the action space
A = DiscreteActionSpace(:a1 => [1,2,3,4], :a2 => [1,2,3,4])

dist = Dict(1 => 0.6, 2 => 0.3, 3=> 0.05, 4 =>0.05)

N = 25

# Define the grammar
grammar = @grammar begin
    R = (R && R) | (R || R) # "and" and "or" expressions for scalar values
    R = all(τ) | any(τ)# τ is true everywhere or τ is eventually true
    τ = (τ .& τ) | (τ .| τ) # "and" and "or" for boolean time series
    τ = _(sample_sym_comparison(A, Symbol(".=="))) # Sample a random equality comparison
end

rn = rand(RuleNode, grammar, :R, 3)
ex = get_executable(rn, grammar)
discrete_actions = sample_series(ex, A, [1.:N...], dist)
length(k)
function convert_actions(mdp, actions, N)
    ks = collect(keys(actions))
    as = Array{Atype}(undef, N)
    for i=1:N
        as[i] = [index_to_action(actions[ks[ki]][i], mdp.models[ki]) for ki in 1:length(ks)]
    end
    as
end

mdp = generate_2car_scene(dt = 0.18)
convert_actions(mdp, discrete_actions, N)




# Define the target rule node:
target_expr(x) = all(x .> 0.75) && any(x .> 0.95) && any(x .<0.8)

N = 25

gp_dist = get_constrained_gp_dist((x,xp) -> exp(-(x-xp)^2/(2*2^2)))

# Define the loss function
function loss(rn::RuleNode, grammar::Grammar)
    ex = get_executable(rn, grammar)
    trials = 10
    total_loss = 0

    for i=1:trials
        actions = []
        try
            actions = sample_series(ex, A, [1.:N...], gp_dist)
        catch e
            return 1e9
        end

        time_series = actions[:x]
        total_loss -= target_expr(time_series)
    end

    total_loss/trials
end

p = GeneticProgram(500,30,6,0.3,0.3,0.4)
results_gp = optimize(p, grammar, :R, loss, verbose = true)

println("loss: ", results_gp.loss, " expression: ", results_gp.expr)
a = sample_series(results_gp.expr, A, [1.:N...], gp_dist)
plot(a[:x])

