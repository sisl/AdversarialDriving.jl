using POMDPSimulators
using Parameters
using LinearAlgebra
using Test
using POMDPs

# Linear model that retains matrices for online fitting
# NOTE: Probably not good to use when the number of paramters is large
@with_kw mutable struct LinearModel
    θ::Array{Float64} = []
    XTX::Array{Float64,2} = []
    XTy::Array{Float64} = []
end

#  Constructors for the linear models
LinearModel(n_dim::Int) = LinearModel(zeros(n_dim), zeros(n_dim, n_dim), zeros(n_dim))
LinearModel(θ::Array{Float64}) = LinearModel(θ, zeros(length(θ), length(θ)), zeros(length(θ)))

# Fit the linear model using new data X, and y added to other data already used to fit the model
function fit!(model::LinearModel, X, y)
    d = length(model.θ)
    # add the new data to the existing data
    model.XTX = model.XTX .+ (X' * X)
    model.XTy = model.XTy .+ (X' * y)

    all(model.XTy .== 0) && return zeros(length(model.θ))

    A = model.XTX + 1e-6*Matrix{Float64}(I,d,d)
    model.θ = pinv(A)*model.XTy
end

# Evaluate the model on some data
forward(model::LinearModel, X) = X * model.θ

# Test the linear model
X = rand(100,2)
X2 = rand(100, 2)
θt = [1., 2.]
y = X * θt
y2 = rand(100)

model = LinearModel(length(θt))
@test model.θ == [0,0]
fit!(model, X, y)
@test all(isapprox.(model.θ, θt, rtol=1e-5, atol=1e-5))
model.θ = θt
@test all(forward(LinearModel([1., 2.]), X) .== y)

two_part_model = LinearModel(length(θt))
fit!(two_part_model, X, y)
fit!(two_part_model, X2, y2)

full_model = LinearModel(length(θt))
fit!(full_model, vcat(X,X2), vcat(y,y2))

@test all(isapprox.(two_part_model.θ, full_model.θ))

mse(est, truth) = sum((est .- truth).^2)/length(truth)
relerr(est, truth) = sum(abs.(est .- truth) ./ max.(truth, est)) / length(truth)

# Definea special type of policy that uses a model, and has a probability of failure
# estimator defined by the user. This estimator will be filled in by the subproblems
@with_kw mutable struct ISPolicy <: Policy
    mdp # The mdp this problem is associated with
    corrective_model # Model that corrects for deviations from the observed probabilty of failure
    estimate # Estimate of the failure probability. Function of the form estimate(mdp, s)
    convert_state = convert_s
    N_actions = :all # Number of actions to sample
end

# Computes the estimated probability of failure at the provided state
# The probability is bounded between 0 and 1, and is
function POMDPs.value(p::ISPolicy, s)
    est = p.estimate(s)
    cor = forward(p.corrective_model, p.convert_state(AbstractArray, s, p.mdp)')
    min(1, max(0, cor + est))
end

# Selects an action to take according to the probability of failure
function action_and_prob(p::ISPolicy, s, rng = Random.GLOBAL_RNG)
    as = actions(p.mdp, s)
    N = length(as)
    p.N_actions != :all && (as = sample(rng, as, p.N_actions, replace=false))
    k = length(as)
    pf = Array{Float64}(undef, k)
    for i=1:k
        a = as[i]
        sp, r = gen(DDNOut((:sp,:r)), p.mdp, s, a, rng)
        pf[i] = action_probability(p.mdp, s, a)*value(p, sp)
    end
    sum_pf = sum(pf)
    pf = (sum_pf == 0) ? ones(k)/k : pf/sum_pf
    ai = rand(Categorical(pf))
    as[ai], pf[ai]*k / N
end

POMDPs.action(p::ISPolicy, s, rng = Random.GLOBAL_RNG) = action_and_prob(p, s, rng)[1]

# Convert an array of states into a state matrix
# NOTE: For some reason using vcat is very slow so we do it this way
function to_mat(S)
    X = Array{Float64, 2}(undef, length(S),length(S[1]))
    for i=1:size(X,1)
        X[i,:] = S[i]
    end
    X
end

# Function for estimating the probability of failure from all the subproblems
function subproblem_estimate_fn(policies, decompose_state, combination_style = :mean)
    function subproblem_estimate(s)
        sub_states = decompose_state(s)
        V = 0
        N_substates = length(sub_states)
        for (i, substate) in sub_states
            val = value(policies[i], substate)
            if combination_style == :mean
                V += val/N_substates
            elseif combination_style == :min
                V = (i==1) ? val : min(val, V)
            elseif combination_style == :max
                V = (i==1) ? val : max(val, V)
            else
                error("unrecognized combination style: ", combination_style)
            end
        end
        V
    end
end

function bellman_residual(mdp, S, V)
    as = actions(mdp)
    residuals = []
    for s in S
        Vlhs = V(s)

        if isterminal(mdp, s)
            push!(residuals, abs(Vlhs - iscollision(mdp, s)))
            continue
        end

        Vrhs = 0
        for a in as
            sp, r = gen(DDNOut((:sp, :r)), mdp, s, a, rng)
            Vrhs += action_probability(mdp, s, a)*V(sp)
        end
        push!(residuals, abs(Vrhs - Vlhs))
    end
    residuals
end

# Simulate the mdp through Neps episodes.
# Requires ISPolicy because it stores the failure probability estimates at each timestep
function sim(policy::ISPolicy, Neps; verbose = true, max_steps = 1000)
    mdp = policy.mdp
    # ρ is the importance sampling weight of the associated action
    # W is the weight of the trajectory from that state (cumulative)
    S, A, R, G, ρ, W, pf_est = [], [], [], [], [], [], []
    total_fails = 0
    for i=1:Neps
        verbose && println("   Rolling out episode ", i)
        s = initialstate(mdp)
        Si, Ai, Ri, ρi, pfi = [policy.convert_state(AbstractArray, s, mdp)], [], [], [], []
        steps = 0
        while !isterminal(mdp, s)
            push!(pfi, policy.estimate(s))
            a, prob = action_and_prob(policy, s)
            push!(Ai, a)
            push!(ρi, action_probability(mdp, s, a) / prob)
            s, r = gen(DDNOut((:sp, :r)), mdp, s, a)
            push!(Si, policy.convert_state(AbstractArray, s, mdp))
            push!(Ri, r)
            steps += 1
            steps >= max_steps && break
        end
        steps >= max_steps && println("Episode timeout at ", max_steps, " steps")
        Gi = reverse(cumsum(reverse(Ri)))
        Wi = reverse(cumprod(reverse(ρi)))
        total_fails += Gi[1] > 0
        push!(S, Si[1:end-1]...)
        push!(A, Ai...)
        push!(R, Ri...)
        push!(G, Gi...)
        push!(ρ, ρi...)
        push!(W, Wi...)
        push!(pf_est, pfi...)
    end
    to_mat(S), A, R, G, ρ, W, pf_est, total_fails
end

# Fits the correct model based on rollouts
function mc_policy_eval!(policy::ISPolicy, iterations, Neps; verbose = true, failure_rate_vec = nothing)
    for iter in 1:iterations
        verbose && println("iteration: ", iter)
        X, _, _, G, _, W, pf_est, failure_rate = sim(policy, Neps, verbose = verbose)
        y = W .* G .- pf_est
        !isnothing(failure_rate_vec) && push!(failure_rate_vec, failure_rate)

        fit!(policy.corrective_model, X, y)
    end
end

