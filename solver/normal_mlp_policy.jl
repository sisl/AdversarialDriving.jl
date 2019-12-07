using Flux: mse, glorot_normal, param, Params
using Distributions
using LinearAlgebra
using Random
include("sampler.jl")

@with_kw mutable struct MLPPolicy
    weights::Dict = Dict()
    input_mean::Array{Float64, 2} = [0.]'
    input_std::Array{Float64, 2} = [1.]'
end

function kl_divergence(d0::MvNormal, d1::MvNormal)
    Σ0 = Matrix(d0.Σ)
    Σ1 = Matrix(d1.Σ)
    Σ1_inv = inv(Σ1)
    dμ = d1.μ .- d0.μ
    k = length(d0.μ)
    0.5*(tr(Σ1_inv * Σ0) + dμ' * Σ1_inv * dμ - k + log(det(Σ1) / det(Σ0)))
end

# Compute the average kl divergence of the two policies π0 and π1
# Given the set of observations
function kl_divergence(π0, π1, obs)
    μ0, Σ0 = forward_nn(π0, obs)
    μ1, Σ1 = forward_nn(π1, obs)

    N = size(μ0, 1)
    tot_kl = 0
    for i=1:N
        d0 = MvNormal(μ0.data[i,:], Σ0.data[1,:])
        d1 = MvNormal(μ1.data[i,:], Σ1.data[1,:])
        tot_kl += kl_divergence(d0, d1)
    end
    tot_kl / N
end

# Converts a dictionary of parameters to Params
to_params(policy) = Params(values(policy.weights))

# Returns the string used to index the weights
W(i) = string("W", i)
b(i) = string("b", i)

# Initialize the policy with the appropriate number of layers/sizes
function init_policy(layers ; init_W = glorot_normal, init_b = zeros, σ_init = ones, estimate_obs_stats = false, N_eps = 100)
    π = MLPPolicy()

    # Add the mlp weights
    for l in 1:length(layers)-1
        in, out = layers[l], layers[l+1]
        π.weights[W(l)] = param(init_W(in, out))
        π.weights[b(l)] = param(init_b(1,out))
    end

    # Add the variance weights
    outputsize = layers[end]
    π.weights["σ2"] = param(σ_init(1,outputsize))

    estimate_obs_stats && ((π.input_mean, π.input_std) = observation_stats(task, π, 100))
    π
end

# Gets the number of layers in the mlp
num_layers(π) = Int64((length(π.weights) - 1) / 2)

# Define the neural network forward to compute mu
function forward_nn(π, input)
    N = num_layers(π)
    x = (input .- π.input_mean ) ./ π.input_std
    for i=1:N-1
        x = x*π.weights[W(i)] .+ π.weights[b(i)]
        x = tanh.(x)
    end
    μ = x*π.weights[W(N)] .+ π.weights[b(N)]
    μ, π.weights["σ2"]
end

# Computes the log probability of observations given a mean and a variance
function log_prob(a, μ, σ2)
    sum((a .- (μ ./ σ2)).^2 .- 0.5 * log.(2*π*σ2), dims=2)
end

# Take a sample action from the network given an observation
function sample_action(π, obs; rng::AbstractRNG = Random.GLOBAL_RNG)
    μ, σ2 = forward_nn(π, obs') # compute the output of the network
    rand(rng, MvNormal(dropdims(μ.data, dims=1), dropdims(σ2.data, dims=1))) # Sample a random action from mean and variance
end

