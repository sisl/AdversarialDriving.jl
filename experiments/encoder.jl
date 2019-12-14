

function kl_divergence(μ, logΣ)
    kl = 0.5 * sum(exp.(log_sigma) .+ μ .^ 2 .- 1 .- log_sigma)
end



