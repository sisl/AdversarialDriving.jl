using POMDPSimulators
using Parameters
using LinearAlgebra
using Test

@with_kw mutable struct LinearModel
    θ::Array{Float64} = []
    XTX::Array{Float64,2} = []
    XTy::Array{Float64} = []
end

LinearModel(n_dim::Int) = LinearModel(zeros(n_dim), zeros(n_dim, n_dim), zeros(n_dim))
LinearModel(θ::Array{Float64}) = LinearModel(θ, zeros(length(θ), length(θ)), zeros(length(θ)))

function fit!(model::LinearModel, X, y)
    d = length(model.θ)
    # add the new data to the existing data
    model.XTX = model.XTX .+ (X' * X)
    model.XTy = model.XTy .+ (X' * y)

    all(model.XTy .== 0) && return zeros(length(model.θ))

    A = model.XTX + 1e-6*Matrix{Float64}(I,d,d)
    model.θ = pinv(A)*model.XTy
end

function forward(model::LinearModel, X)
    return X * model.θ
end

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





update_policy!(p, model) = nothing

function to_mat(S)
    X = Array{Float64, 2}(undef, length(S),length(S[1]))
    for i=1:size(X,1)
        X[i,:] = S[i]
    end
    X
end

function sim(mdp, policy, Neps)
    # ρ is the importance sampling weight of the associated action
    # W is the weight of the trajectory from that state (cumulative)
    S, A, R, G, ρ, W = [], [], [], [], [], []
    for i=1:Neps
        s = rand(initialstate_distribution(mdp))
        Si, Ai, Ri, ρi = [convert_s(mdp, s)], [], [], []
        while !isterminal(mdp, s)
            a, prob = action(policy, s)
            push!(Ai, a)
            push!(ρi, action_probability(mdp, s, a) / prob)
            s, r = gen(DDNOut((:sp, :r)), mdp, s, a)
            push!(Si, convert_s(mdp, s))
            push!(Ri, r)
        end
        Gi = reverse(cumsum(reverse(Ri)))
        Wi = reverse(cumprod(reverse(ρi)))
        push!(S, Si[1:end-1]...)
        push!(A, Ai...)
        push!(R, Ri...)
        push!(G, Gi...)
        push!(ρ, ρi...)
        push!(W, Wi...)
    end
    to_mat(S), A, R, G, ρ, W
end

function mc_policy_eval(mdp, policy, model, max_iterations, Neps)
    for iter in 1:max_iterations
        println("iteration: ", iter)
        X, _, _, G, _, W = sim(mdp, policy, Neps)
        y = W .* G

        println("X size: ", size(X), " ysize: ", size(y))
        fit!(model, X, y)
        update_policy!(policy, model)
    end
    model
end

