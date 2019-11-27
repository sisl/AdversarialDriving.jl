include("../solver/normal_mlp_policy.jl")
using Test

# Test the parameter naming functions
@test W(1) == "W1"
@test b(10) == "b10"

# Initialize a policy
p = init_policy([2,10,3], init_W = ones)
@test haskey(p, "W1") && haskey(p, "W2")
@test haskey(p, "b1") && haskey(p, "b2")
@test haskey(p, "σ2")
@test all(p["W1"] .== 1) && all(p["W2"] .== 1)
@test all(p["b1"] .== 0) && all(p["b2"] .== 0)
@test all(p["σ2"] .== 1)
@test size(p["W1"]) == (2,10) && size(p["W2"]) == (10,3)
@test size(p["b1"]) == (1,10) && size(p["b2"]) == (1,3)
@test size(p["σ2"]) == (1,3)

# Check that we can compute the correct number of layers
@test num_layers(p) == 2

# Check that we can compute the forward pass properly
μ, σ2 = forward_nn(p, zeros(100,2))
@test μ == zeros(100,3)
@test σ2 == [1,1,1]'

# Get that log probabilities of some actions
logprobs = log_prob(zeros(100,3), μ, σ2)
@test size(logprobs) == (100,1)
@test all(logprobs .== 3*log(pdf(Normal(0,1), 0)))

# Check that we can sample an action
as = sample_action(p, [0.,0.])
@test size(as) == (3,)

