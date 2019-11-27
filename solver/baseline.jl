include("sampler.jl")
using LinearAlgebra

# Featurizes the batch's returns and times
function features(batch::Batch)
    hcat(batch.observations, batch.observations .^ 2, batch.times, batch.times .^ 2, batch.times .^ 3, ones(batch.N))
end

# Fits a linear model to the featurized batch
# λ is the regularization constant
# Returns a vector of weights to compute return from a featurized observation
function fit(batch::Batch, regularization_coeff::Float64)
    X = features(batch)
    y = batch.returns
    n,d = size(X)
    A = X' * X ./ n + regularization_coeff*Matrix{Float64}(I,d,d)
    B = X' * y ./ n
    pinv(A)*B
end

# Compute the predicted returns for the provided observations and weights
linear_forward(batch::Batch, weights::Array{Float64}) = features(batch) * weights

# weights are the linear weights for the baseline model
# γ is the discount factor of the POMDP
# λ is the tunable parameter describing the λ-return
function gae(batch::Batch, weights::Array{Float64}, γ::Float64, λ::Float64)
    values = linear_forward(batch, weights)

    # Compute next-state values, zeroing out the state after end of ep
    next_values = vcat(values[2:end], 0)
    next_values[batch.episode_ends] .= 0

    deltas = batch.rewards .+ γ * values .- next_values
    gae = 0
    advantages = zeros(size(deltas))
    for i=length(deltas):-1:1
        gae = gae * γ * λ + deltas[i]
        advantages[i] = gae
    end
    advantages
    #TODO: should we be normalizing the advantages?
end

