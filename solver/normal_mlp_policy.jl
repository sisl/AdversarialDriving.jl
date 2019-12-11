using Flux: mse, glorot_normal, param, Params
using Distributions
using LinearAlgebra
using Random
using Serialization
include("sampler.jl")

save_policy(name, policy) = serialize(string(name, ".model"), policy)
load_policy(name) = deserialize(string(name, ".model"))

@with_kw mutable struct MLPPolicy
    weights::Dict = Dict()
end

Base.:(==)(l::MLPPolicy, r::MLPPolicy) = (l.weights == r.weights)

function kl_divergence(d0::MvNormal, d1::MvNormal)
    Σ0 = Matrix(d0.Σ)
    Σ1 = Matrix(d1.Σ)
    Σ1_inv = inv(Σ1)
    dμ = d1.μ .- d0.μ
    k = length(d0.μ)
    0.5*(tr(Σ1_inv * Σ0) + dμ' * Σ1_inv * dμ - k + log(det(Σ1) / det(Σ0)))
end

# Compute the average kl divergence of the two policies p0 and p1
# Given the set of observations
function kl_divergence(p0, p1, obs)
    μ0, Σ0 = forward_nn(p0, obs)
    μ1, Σ1 = forward_nn(p1, obs)

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
function init_policy(layers ; init_W = glorot_normal, init_b = zeros, σ_init = ones, estimate_obs_stats = false, N_eps = 100, task = nothing)
    p = MLPPolicy()

    # Add the mlp weights
    for l in 1:length(layers)-1
        in, out = layers[l], layers[l+1]
        p.weights[W(l)] = param(init_W(in, out))
        p.weights[b(l)] = param(init_b(1,out))
    end

    # Add the variance weights
    outputsize = layers[end]
    p.weights["σ2"] = param(σ_init(1,outputsize))
    p
end

# Gets the number of layers in the mlp
num_layers(p) = Int64((length(p.weights) - 1) / 2)

# Define the neural network forward to compute mu
function forward_nn(p, input)
    N = num_layers(p)
    x = input
    for i=1:N-1
        x = x*p.weights[W(i)] .+ p.weights[b(i)]
        x = tanh.(x)
    end
    μ = x*p.weights[W(N)] .+ p.weights[b(N)]
    μ, p.weights["σ2"]
end

# Computes the log probability of observations given a mean and a variance
function log_prob(a, μ, σ2)
    sum((a .- (μ ./ σ2)).^2 .- 0.5 * log.(6.2831853071794*σ2), dims=2)
end

# Take a sample action from the network given an observation
function sample_action(p, obs; rng::AbstractRNG = Random.GLOBAL_RNG)
    μ, σ2 = forward_nn(p, obs') # compute the output of the network
    rand(rng, MvNormal(dropdims(μ.data, dims=1), dropdims(σ2.data, dims=1))) # Sample a random action from mean and variance
end

