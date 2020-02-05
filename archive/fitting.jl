using Parameters
using LinearAlgebra
using Test
using Flux: mse, glorot_normal, params, relu, Chain, Dense,σ, softmax, sigmoid, @epochs, mse, ADAM, train!

@with_kw mutable struct LinearModel
    θ::Array{Float64} = []
end

LinearModel(n_dim::Int) = LinearModel(zeros(n_dim))

function fit!(model::LinearModel, X, y)
    n,d = size(X)
    if all(y .== 0)
        return zeros(d)
    end

    A = X' * X ./ n + 1e-6*Matrix{Float64}(I,d,d)
    B = X' * y ./ n
    model.θ = pinv(A)*B
end

function forward(model::LinearModel, X)
    return X * model.θ
end


# Test the linear model
X = rand(100,2)
θt = [1., 2.]
y = X * θt

model = LinearModel(length(θt))
@test model.θ == [0,0]
fit!(model, X, y)
@test all(isapprox.(model.θ, θt, rtol=1e-5, atol=1e-5))
model.θ = θt
@test all(forward(LinearModel([1., 2.]), X) .== y)



# Initialize the policy with the appropriate number of layers/sizes
function init_MLP(layers; initW = glorot_normal, initb = zeros)
    N = length(layers)
    a1 = [Dense(layers[l], layers[l+1], relu, initW = initW, initb = initb) for l in 1:N-2]
    a2 = Dense(layers[N-1], layers[N], identity, initW = initW, initb = initb)
    Chain( a1..., a2 )
end
to_data(X, y) = [(X[i,:], y[i]) for i in 1:size(X,1)]

function fit!(model::Chain, X, y, N_iterations = 1)
    loss(xx, yy) = mse(model(xx), yy)
    ps = params(model)

    # later
    opt = ADAM(0.01, (0.9, 0.999))
    for i=1:N_iterations
        train!(loss, ps, to_data(X,y), opt)
    end
end

# Define the neural network forward to compute mu
forward(model::Chain, X) = dropdims(model(X'), dims=1)

# Test the Neural Network model
X = rand(100,5)
θt = [2., 2., 5, 6, 7]
y = X * θt
y = y./maximum(y)

model = init_MLP([5, 1])
l1 = sum((forward(model, X) .- y).^2)
fit!(model, X, y, 10)
l2 = sum((forward(model, X) .- y).^2)
params(model)

@test 10*l2 < l1

