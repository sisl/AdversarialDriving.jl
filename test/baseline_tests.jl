include("../solver/baseline.jl")
using Test

N, o, a = 50, 3, 6
batch = Batch(2*N, o, a)
append_ep!(batch, 3*ones(N,o), ones(N,a), ones(N), cumsum(ones(N)), cumsum(ones(N)))
append_ep!(batch, 3*ones(N,o), ones(N,a), ones(N), cumsum(ones(N)), cumsum(ones(N)))

# Check that the featurization is correct
@test all(features(batch)[2,:] .== [3.,3.,3.,9.,9.,9.,2.,4.,8.,1.])
@test size(features(batch)) == (2*N, 2*o+4)

# Check that the normal equation is working properly
batch.observations = rand(2*N,o)
w = fit(batch, 0.)
X = features(batch)
@test all(isapprox.(X*w, batch.returns)) # This works because returns == times

# Make sure the forward pass is doing the correct matrix multiply
@test linear_forward(batch, w) == X*w

# Compute the gae
advantages = gae(batch, w, 1., 1.)
@test size(advantages) == (2*N,)

