using Flux: mse, glorot_normal, param
using Flux.Tracker: update!,

# Converts a dictionary of parameters to Params
function to_params(weights)
    params = Params()
    for (k,v) in weights
        push!(params, v)
    end
    return params
end


# define the neural network forward
function forward_nn(weights, input)
    z1 = weights["W1"]*input .+ weights["b1"]
    o = weights["W2"]*relu.(z1) .+ weights["b2"]
    # softmax(o)
end

# initialize the weights
weights = Dict()
weights["W1"] = Flux.param(glorot_normal(10,2))
weights["b1"] = Flux.param(zeros(10))

weights["W2"] = Flux.param(glorot_normal(2,10))
weights["b2"] = Flux.param(zeros(2))


B = 100

function loss(x, y)
    grads = Tracker.gradient(()->mse(forward_nn(weights, x),y), to_params(weights), nest = true)

    new_weights = Dict()
    for (k,p) in weights
        new_weights[k] = p - 0.1*grads[p]
    end

    mse(forward_nn(new_weights, x), y)
end



for i=1:1000
    x = rand([0,1],2,B)
    y = hcat(Float64.(x[1,:] .== x[2,:]), Float64.(x[1,:] .!= x[2,:]))'
    println(loss(x,y))

    grads = Tracker.gradient(()->loss(x,y), to_params(weights))

    for (k,p) in weights
        Flux.Tracker.update!(opt, p, grads[p])
    end
end




