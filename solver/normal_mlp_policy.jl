using Flux: mse, glorot_normal, param
using Flux.Tracker: update!,
using Distributions

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
function init_policy(layers; init_w = glorot_normal, init_b = zeros, σ_init = ones)
    weights = Dict()

    # Add the mlp weights
    for i in 1:length(layers)-1
        in, out = layers[i], layers[i+1]
        weights[W(l)] = Flux.param(init_w(out, in))
        weights[b(l)] = Flux.param(init_b(out))
    end

    # Add the variance weights
    outputsize = layers[end]
    weights["σ2"] = Flux.param(σ_init(outputsize))

    weights
end

# Gets the number of layers in the mlp
num_layers(weights) = (length(weights) - 1 / 2)

# Define the neural network forward to compute mu
function forward_nn(weights, input)
    N = num_layers(weights)
    x = input
    for i=1:N-1
        x = weights[W(i)]*x .+ weights[b(i)]
        x = tanh.(x)
    end
    μ = weights[W(N)]*x .+ weights[b(N)]
    μ, weights["σ2"]
end

# Computes the log probability of observations given a mean and a variance
function log_prob(obs, μ, σ2)
    sum((obs .- (μ ./ σ2)).^2 .- 0.5 * log.(2*π*σ2), dims=1)
end

# Take a sample action from the network given an observation
function sample_action(weights, obs; rng::AbstractRNG = Random.GLOBAL_RNG)
    μ, σ2 = forward_nn(weights, obs) # compute the output of the network
    rand(rng, MvNormal(μ, σ2)) # Sample a random action from mean and variance
end
