using Flux
using Flux: mse, glorot_normal, param, Params
using Flux.Tracker: update!
using Distributions
using Random

# Converts a dictionary of parameters to Params
function to_params(weights)
    params = Params()
    for (k,v) in weights
        push!(params, v)
    end
    return params
end

# Returns the string used to index the weights
W(i) = string("W", i)
b(i) = string("b", i)

# Initialize the policy with the appropriate number of layers/sizes
function init_policy(layers; init_W = glorot_normal, init_b = zeros, σ_init = ones)
    weights = Dict()

    # Add the mlp weights
    for l in 1:length(layers)-1
        in, out = layers[l], layers[l+1]
        weights[W(l)] = Flux.param(init_W(in, out))
        weights[b(l)] = Flux.param(init_b(1,out))
    end

    # Add the variance weights
    outputsize = layers[end]
    weights["σ2"] = Flux.param(σ_init(1,outputsize))

    weights
end

# Gets the number of layers in the mlp
num_layers(weights) = Int64((length(weights) - 1) / 2)

# Define the neural network forward to compute mu
function forward_nn(weights, input)
    N = num_layers(weights)
    x = input
    for i=1:N-1
        x = x*weights[W(i)] .+ weights[b(i)]
        x = tanh.(x)
    end
    μ = x*weights[W(N)] .+ weights[b(N)]
    μ, weights["σ2"]
end

# Computes the log probability of observations given a mean and a variance
function log_prob(a, μ, σ2)
    sum((a .- (μ ./ σ2)).^2 .- 0.5 * log.(2*π*σ2), dims=2)
end

# Take a sample action from the network given an observation
function sample_action(weights, obs; rng::AbstractRNG = Random.GLOBAL_RNG)
    μ, σ2 = forward_nn(weights, obs') # compute the output of the network
    rand(rng, MvNormal(dropdims(μ.data, dims=1), dropdims(σ2.data, dims=1))) # Sample a random action from mean and variance
end

