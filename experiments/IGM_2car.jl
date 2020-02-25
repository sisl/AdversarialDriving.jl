using ExprOptimization
using ExprRules
using POMDPSimulators
using POMDPPolicies
include("../../ltl_sampling/LTLSampling.jl")
include("../simulator/adm_task_generator.jl")
include("plot_utils.jl")
include("../solver/is_probability_estimator.jl")
using Plots; gr()
rng = Random.GLOBAL_RNG

# Define the action space
A = DiscreteActionSpace(:a1 => [1,2,3,4,5,6,7])

# Define the probability distribution
dist = Dict(1 => 1 - (4e-3 + 2e-2), 2 => 1e-3, 3=> 1e-2, 4 =>1e-2, 5=>1e-3, 6=>1e-3, 7=>1e-3)

# Define the numer of interesting timesteps
N = 20

# Create the mdp
# mdp = generate_2car_scene(dt = 0.18, ego_s = 15., ego_v = 9., other_s = 29., other_v = 10.) # Speedup failure
# mdp = generate_2car_scene(dt = 0.18, ego_s = 15., ego_v = 9., other_s = 29., other_v = 20.) # Blinker failure
mdp = generate_2car_scene(dt = 0.18, ego_s = 19., ego_v = 9., other_s = 43., other_v = 29.) # change speed and Blinker failure

h = POMDPSimulators.simulate(HistoryRecorder(rng = rng), mdp, FunctionPolicy((s) -> random_action(mdp, s, rng)))
make_interact(state_hist(h), mdp.models, mdp.roadway, egoid=2)

# Define the grammar
grammar = @grammar begin
    R = (R && R)| (R || R) # "and" and "or" expressions for scalar values
    R = all_before(τ, C) | all_after(τ, C) | all_between(τ, C, C)
    C = |(1:20)
    τ = (τ .& τ) | (τ .| τ) # "and" and "or" for boolean time series
    τ = _(sample_sym_comparison(A, Symbol(".=="))) # Sample a random equality comparison
end

rn = rand(RuleNode, grammar, :R, 4)
ex = get_executable(rn, grammar)
# ex = Meta.parse("all((a1 .== 4) .| (a1 .== 5))")
discrete_actions = sample_series(ex, A, [1:N...], dist)
aseq = convert_actions(mdp, discrete_actions, N)
state_h, r = fixed_action_rollout(mdp, aseq)
make_interact(state_h, mdp.models, mdp.roadway, egoid=2)

function convert_actions(mdp, actions, N)
    ks = collect(keys(actions))
    as = Array{Atype}(undef, N)
    for i=1:N
        as[i] = [index_to_action(actions[ks[ki]][i], mdp.models[ki]) for ki in 1:length(ks)]
    end
    as
end

function loss(rn::RuleNode, grammar::Grammar)

    # Get the executable
    ex = get_executable(rn, grammar)
    trials, total_loss = 10, 0
    λ = 1/trials
    for i=1:trials
        discrete_actions = []
        try
            discrete_actions = sample_series(ex, A, [1:N...], dist)
        catch e
            return 1e9
        end
        aseq = convert_actions(mdp, discrete_actions, N)
        _, prob, r = fixed_action_rollout(mdp, aseq)
        # total_loss -= r - λ*prob
        total_loss -= r*prob
    end
    total_loss/trials
end


# Optimize:
p = GeneticProgram(2000,40,10,0.3,0.3,0.4)
results_gp = optimize(p, grammar, :R, loss, verbose = true)
post_prune = get_executable(prune(results_gp.tree, :R, grammar), grammar)
println("loss: ", results_gp.loss, " expression (pre-prune): ", results_gp.expr, " expression(pos-prune): ", post_prune)

# Show an example of the failure in interactive format
a = sample_series(results_gp.expr, A, [1.:N...], dist)
aseq = convert_actions(mdp, a, N)
state_h, prob, r = fixed_action_rollout(mdp, aseq)
make_interact(state_h, mdp.models, mdp.roadway, egoid=2)
write_scenes(state_h, mdp.models, mdp.roadway, "2car_res3_frame", egoid=2)

# Compute and compare probability of action
gp_failure_avg_prob = []
is_failure_avg_prob = []
for i=1:100
    # print("i=$i, GP....")
    # Fill the average probability of the GP expression
    while true
        a = sample_series(results_gp.expr, A, [1.:N...], dist)
        aseq = convert_actions(mdp, a, N)
        state_hist, prob, r = fixed_action_rollout(mdp, aseq)
        if r>0
            push!(gp_failure_avg_prob, prob)
            break
        end
    end
    # println("Is...")
    timeout = 1
    thresh = 1e3
    while true && timeout < thresh
        timeout += 1
        h = POMDPSimulators.simulate(HistoryRecorder(rng = rng), mdp, UniformISPolicy(mdp, Random.GLOBAL_RNG))
        rs = reward_hist(h)
        if sum(rs) > 0
            ss = POMDPSimulators.state_hist(h)
            as = collect(POMDPSimulators.action_hist(h))
            prob = 0
            for i=1:length(as)
                prob += true_action_probability(mdp, ss[i], as[i])
            end
            push!(is_failure_avg_prob, prob / length(as))
            break
        end
    end
    if timeout >= thresh
        println("importance sampling timeout")
    end
end


println("Genetic Programmed expression average action probability: ", mean(gp_failure_avg_prob), " +/- ", std(gp_failure_avg_prob))
println("Uniform IS average action probability: ", mean(is_failure_avg_prob), " +/- ", std(is_failure_avg_prob))

# Fraction of failures (per 100)
Nsamples = 100
Ntrials = 5
function run_trials(mdp, pol, Nsamps, Ntrials, rng)
    results = [sum([POMDPSimulators.simulate(RolloutSimulator(rng = rng), mdp, pol) for i=1:Nsamps]) for i=1:Ntrials]
    mean(results), std(results)
end

function run_trials_gp(mdp, expr, Nsamps, Ntrials, rng)
    results = []
    for i=1:Ntrials
        tot_r = 0
        for i=1:Nsamples
            a = sample_series(results_gp.expr, A, [1:N...], dist)
            aseq = convert_actions(mdp, a, N)
            state_h, prob, r = fixed_action_rollout(mdp, aseq)
            tot_r += r
        end
        push!(results, tot_r)
    end
    mean(results), std(results)
end

# run_trials(mdp, FunctionPolicy((s)->random_action(mdp, s, rng)), Nsamples, Ntrials, rng)
mean_gp, std_gp = run_trials_gp(mdp, results_gp.expr, Nsamples, Ntrials, rng)
mean_is, std_is  = run_trials(mdp, UniformISPolicy(mdp, rng), Nsamples, Ntrials, rng)
println("gp mean: ", mean_gp, " gp_std: ", std_gp)
println("is mean: ", mean_is, " is_std: ", std_is)

