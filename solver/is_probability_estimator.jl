using Statistics
struct UniformISPolicy <: Policy
    mdp
    rng
end

POMDPs.action(p::UniformISPolicy, s) = rand(rng, actions(mdp, s))

function action_and_prob(p::UniformISPolicy, s, rng)
    as = actions(mdp,s)
    a = rand(rng, as)
    a, 1. / length(as)
end

function is_rollout(mdp, policy, rng)
    s = initialstate(mdp)
    tot_r = 0
    tot_prob = 1
    while !isterminal(mdp, s)
        a, prob = action_and_prob(policy, s, rng)
        tot_prob *= true_action_probability(mdp, s, a) / prob
        s, r = gen(DDNOut((:sp,:r)), mdp, s, a, rng)
        tot_r += r
    end
    tot_r, tot_prob
end


function compute_mean(Nsamples, mdp, policy, rng)
    samps = []
    running_mean = Array{Float64}(undef, Nsamples)
    running_var = Array{Float64}(undef, Nsamples)
    running_var[1] = 0
    for i =1:Nsamples
        f, w = is_rollout(mdp, policy, rng)

        push!(samps, f*w)

        running_mean[i] = mean(samps)
        if i > 1
            running_var[i] = std(samps) / sqrt(i)
        end
    end
    running_mean, running_var
end

# function compute_mean_test(Nsamples, μ)
#     running_mean = Array{Float64}(undef, Nsamples)
#     tot = 0
#     for i =1:Nsamples
#         f, w = sample_is(μ)
#         tot += f*w
#         running_mean[i] = tot / i
#     end
#     running_mean
# end
#
# function sample_is(μ)
#     v = rand(Normal(μ))
#     f = v > 0.5
#     w = exp(logpdf(Normal(), v)) / exp(logpdf(Normal(μ), v))
#     return f, w
# end



