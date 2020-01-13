using Flux
using Flux: glorot_normal, onehotbatch
using Flux.Tracker: update!
using Distributions
using Random
using POMDPs
using POMDPModels
using POMDPModelTools
using DiscreteValueIteration
using Random

function init_Qnet(layers; init_W = glorot_normal, init_b = zeros)
    Q = Dict()
    for l in 1:length(layers)-1
        in, out = layers[l], layers[l+1]
        Q[string("W",l)] = param(init_W(in, out))
        Q[string("b",l)] = param(init_b(1,out))
    end
    Q
end

function forward(Q, s)
    N = convert(Int, length(Q) / 2)
    for i=1:N-1
        s = relu.(s*Q[string("W",i)] .+ Q[string("b",i)])
    end
    s*Q[string("W",N)] .+ Q[string("b", N)]
end

function sample_all_states(pomdp; rng::AbstractRNG = Random.GLOBAL_RNG)
    N = length(states(pomdp))*length(actions(pomdp))
    ns = length(initialstate(pomdp,rng))

    S, Sp = Array{Float64,2}(undef, N, ns), Array{Float64,2}(undef, N, ns)
    A, R = Vector{Int64}(undef, N), Vector{Float64}(undef, N)
    isT = Vector{Bool}(undef, N)
    i = 1
    for s in states(pomdp)
        for a in actions(pomdp)
            isT[i] = isterminal(g,s)
            S[i, :] .= s
            A[i] = actionindex(pomdp, a)
            sp, r = gen(pomdp, s, a, rng)
            Sp[i, :] .= sp
            R[i] = r
            i = i+1
        end
    end
    S, A, Sp, R, isT
end

# This function solves the problem for the max
function Q_Loss(pomdp, Qnet, B = 1)
    S, A, Sp, R, isT  = sample_all_states(pomdp)
    Qval = sum(forward(Qnet, S).*(onehotbatch(A, [1,2,3,4])'), dims = 2)
    target = R .+ (.!isT) .* (discount(pomdp)*dropdims(maximum(forward(Qnet, Sp), dims=2), dims=2))
    loss = sum((target.data .- Qval).^2)
end

# this functino solves the problem for
function Qp_Loss(pomdp, Qnet, B = 1)
    S, A, Sp, R, isT  = sample_all_states(pomdp)
    Qpval = sum(forward(Qnet, S).*(onehotbatch(A, [1,2,3,4])'), dims = 2)
    target = R .+ discount(pomdp)* (.!isT) .* forward(Qnet, Sp)
    loss = sum((target.data .- Qval).^2)
end

g = gridworld(3,3, size = (3,3))
layers = [2, 32, 32, length(actions(g))]
Qnet = init_Qnet(layers)
discount(g)
Q_Loss(g, Qnet)

opt = ADAM(0.001, (0.9, 0.999))

θ = Params(values(Qnet))
for i=1:1000
    grads = Tracker.gradient(() -> Q_Loss(g, Qnet), θ)
    for p in values(Qnet)
      update!(opt, p, grads[p])
      println("loss: ", Q_Loss(g, Qnet))
    end
end

forward(Qnet, [1,1]')


actions(g)

