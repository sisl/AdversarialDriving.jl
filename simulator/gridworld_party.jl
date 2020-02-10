using POMDPs
using POMDPModelTools
using StaticArrays
using Parameters
using Base.Cartesian
using Random
using Test
using Compose
using ColorSchemes

const GWPos = SVector{2,Int}
const dir = Dict(:up=>GWPos(0,1), :down=>GWPos(0,-1), :left=>GWPos(-1,0), :right=>GWPos(1,0))
const aind = Dict(:up=>1, :down=>2, :left=>3, :right=>4)
const syma = [:up, :down, :left, :right]

# Gridworld Party MDP. Multiple agents trying to reach a goal on a gridworld
@with_kw mutable struct GridworldParty <:MDP{Vector{GWPos}, Vector{Symbol}}
    n_agents::Int = 1
    size::Tuple{Int, Int} = (10,10)
    goals::Vector{GWPos} = []
    tprob::Float64 = 0.7
    discount::Float64 = 0.95
    reward_type::Symbol = :normal # use :normal or :adversarial
    actions::Vector{Vector{Symbol}} = [[syma[i] for i in index.I] for index in CartesianIndices(Tuple([4 for _ in 1:n_agents]))][:]
    actionindex::Dict{Vector{Symbol}, Int} = Dict(a => convert_I(n_agents, a) for a in actions)
    a_dict = nothing
end

AdversarialGridworldParty(mdp::GridworldParty) = GridworldParty(n_agents = mdp.n_agents, size = mdp.size, goals = mdp.goals, tprob = mdp.tprob, discount = 1.0, reward_type = :adversarial,)

# Gets the pairwise combination of indices
# 3 agents would be [[1,2], [1,3], [2,3]]
function decompose_indices(n_agents)
    indices = []
    for ij in CartesianIndices((n_agents, n_agents))
        if ij.I[1] != ij.I[2] && ij.I[1] < ij.I[2]
            push!(indices, [ij.I[1], ij.I[2]])
        end
    end
    indices
end

# Decompse the state into the subproblem states
function decompose_state(s::Vector{GWPos})
    indices = decompose_indices(length(s))
    Dict(i => s[indices[i]] for i in 1:length(indices))
end

# Decompose the mdp into sub-problem mdps
function decompose(mdp::GridworldParty)
    indices = decompose_indices(mdp.n_agents)
    mdps = Array{GridworldParty}(undef, 0)
    for i in indices
        push!(mdps, GridworldParty(n_agents = 2, goals = mdp.goals[i],
                                size = mdp.size,
                                tprob = mdp.tprob,
                                discount = mdp.discount,
                                reward_type = mdp.reward_type))
    end
    mdps
end

POMDPs.initialstate(mdp::GridworldParty, rng::AbstractRNG = Random.GLOBAL_RNG) = [mdp.goals[2:end]..., mdp.goals[1]]

function POMDPs.states(mdp::GridworldParty)
    nstates = (mdp.size[1]*mdp.size[2])^mdp.n_agents
    states = Vector{Vector{GWPos}}(undef, nstates)
    index = 1
    for ci in CartesianIndices(Tuple(vcat([[mdp.size[1], mdp.size[2]] for i in 1:mdp.n_agents]...)))
        states[index] = convert_s(Vector{GWPos}, [Float64.(ci.I)...], mdp)
        index += 1
    end
    states
end


# Convert back and forth between vector representations
POMDPs.convert_s(::Type{AbstractArray}, s::Vector{GWPos}, mdp::GridworldParty) = vcat([Float64.(pos) for pos in s]...)
POMDPs.convert_s(::Type{Vector{Float64}}, s::Vector{GWPos}, mdp::GridworldParty) = vcat([Float64.(pos) for pos in s]...)
POMDPs.convert_s(::Type{Vector{GWPos}}, s::AbstractArray{Float64}, mdp::GridworldParty) = [GWPos(s[2*i - 1], s[2*i]) for i in 1:mdp.n_agents]

# Each agent can choose one of four actions each step (up, down, left, right)
POMDPs.actions(mdp::GridworldParty) = mdp.actions
POMDPs.actionindex(mdp::GridworldParty, a::Vector{Symbol}) = mdp.actionindex[a]

function convert_I(n_agents::Int, a::Vector{Symbol})
    ind = [aind[ai] for ai in a]
    sum((ind .- 1) .* [4^i for i = 0:n_agents-1]) + 1
end

function action_probability(mdp::GridworldParty, s::Vector{GWPos}, a::Vector{Symbol})
    tot_prob = 1.0
    a_would = mdp.a_dict[s]
    alt_prob = (1.0 - mdp.tprob) / 3.
    for i = 1:mdp.n_agents
        tot_prob *= (a_would[i] == a[i] ? mdp.tprob : alt_prob)
    end
    tot_prob
end

action_dict(mdp::GridworldParty, policy) = Dict(s => action(policy, s) for s in states(mdp))

# Do any agents overlap?
any_overlap(mdp::GridworldParty, s::Vector{GWPos}) = any([s[i1] == s[i2] && i1 > i2 for i1 in 1:mdp.n_agents, i2 in 1:mdp.n_agents])
all_overlap(mdp::GridworldParty, s::Vector{GWPos}) = all([s[i] == s[1] for i in 1:mdp.n_agents])

# Have all the agents reached their goal?
all_goal(mdp::GridworldParty, s::Vector{GWPos}) = all([s[i] == mdp.goals[i] for i=1:mdp.n_agents])
any_goal(mdp::GridworldParty, s::Vector{GWPos}) = any([s[i] == mdp.goals[i] for i=1:mdp.n_agents])

# Whether the scene is terminal or not
POMDPs.isterminal(mdp::GridworldParty, s::Vector{GWPos}) = any(s .== [GWPos(-1,-1)])
POMDPs.discount(mdp::GridworldParty) = mdp.discount

# Snaps positions to gridworld boundary
snap_to_boundary(sz::Tuple{Int,Int}, pos::GWPos) = GWPos(min.(sz, max.(1, pos)))

# Stochastically moves once agent based on its action and the randomness of the world
function move(mdp::GridworldParty, s::GWPos, a::Symbol, rng::AbstractRNG = Random.GLOBAL_RNG)
    if mdp.reward_type == :adversarial || rand(rng) < mdp.tprob
        sp = s + dir[a]
    else
        sp = s + dir[rand(rng, syma[a .!= syma])]
    end
    snap_to_boundary(mdp.size, sp)
end


# Returns a sample next state and reward
function POMDPs.gen(mdp::GridworldParty, s::Vector{GWPos}, a::Vector{Symbol}, rng::AbstractRNG = Random.GLOBAL_RNG)
    if isterminal(mdp, s)
        sp = s
    elseif any_goal(mdp, s) || all_overlap(mdp, s)
        sp = [GWPos(-1,-1) for i=1:mdp.n_agents]
    else
        sp = [move(mdp, s[i], a[i], rng) for i in 1:mdp.n_agents]
    end
    r = reward(mdp, s)

    (sp=sp, r=r)
end
POMDPs.gen(::DDNOut{(:sp, :r)}, mdp::GridworldParty, s::Vector{GWPos}, a::Vector{Symbol}, rng::AbstractRNG = Random.GLOBAL_RNG) = gen(mdp, s, a, rng)

# Returns the reward for the provided state
function POMDPs.reward(mdp::GridworldParty, s::Vector{GWPos})
    if isterminal(mdp, s)
        return 0
    else
        mdp.reward_type == :normal ? Float64(any_goal(mdp, s)) : Float64(all_overlap(mdp, s))
    end
end

# Renders the gridworld
function POMDPModelTools.render(mdp::GridworldParty, s::Vector{GWPos}; agent_colors = ["blue", "orange", "purple", "yellow"])
    nx, ny = mdp.size
    cells = []
    for x in 1:nx, y in 1:ny
        index = findfirst([GWPos(x,y)] .== mdp.goals)
        ctx = context((x-1)/nx, (ny-y)/ny, 1/nx, 1/ny)
        cell = compose(ctx, rectangle(), fill(!isnothing(index) ? agent_colors[index] : "white"))
        push!(cells, cell)
    end
    grid = compose(context(), Compose.stroke("gray"), cells...)
    outline = compose(context(), rectangle())

    i = 1
    agents = []
    for pos in s
        x,y = pos
        agent_ctx = context((x-1)/nx, (ny-y)/ny, 1/nx, 1/ny)
        agent = compose(agent_ctx, circle(0.5, 0.5, 0.4), Compose.stroke("black"), fill(agent_colors[i]))
        push!(agents, agent)
        i += 1
    end
    agents_comp = compose(context(), agents...)

    sz = min(w,h)
    return compose(context((w-sz)/2, (h-sz)/2, sz, sz), agents_comp, grid, outline)
end

###### Testing begins here ##########
mdp = GridworldParty(n_agents = 3, goals = [(1,1), (10,10), (1,10)])
adv_mdp = GridworldParty(n_agents = 3, goals = [(1,1), (10,10), (1,10)], reward_type = :adversarial)
mdp2 = GridworldParty(n_agents = 2, goals = [(1,1), (10,10)])
s = [GWPos(1,1), GWPos(10,10), GWPos(1,10)]
s2 = [GWPos(9,9), GWPos(9,9), GWPos(9,9)]
s3 = [GWPos(-1,-1), GWPos(-1,-1), GWPos(-1,-1)]
s4 = [GWPos(1,1), GWPos(5,5)]
s5 = [GWPos(5,5), GWPos(5,5)]
s6 = [GWPos(-1,-1), GWPos(-1,-1)]

@test !any_overlap(mdp, s) && any_overlap(mdp, s2) && all_overlap(mdp, s2) && !all_overlap(mdp, s)
@test all_overlap(mdp2, s5) && !any_overlap(mdp2, s4)
@test all_goal(mdp, s) && !all_goal(mdp, s2) && any_goal(mdp, s) && !any_goal(mdp, s2)
@test !isterminal(mdp, s) && !isterminal(mdp, s2) && isterminal(mdp, s3)
@test length(actions(mdp)) == 4^mdp.n_agents
@test actionindex(mdp, [:up, :up, :up]) == 1
@test reward(mdp, s) == 1 && reward(mdp, s2) == 0
@test reward(adv_mdp, s) == 0 && reward(adv_mdp, s2) == 1
@test snap_to_boundary((10,10), GWPos(0,11)) == GWPos(1,10)

rng = MersenneTwister(0)
t = 0
for i=1:100
    global t += move(mdp, GWPos(1,1), :right, rng) == GWPos(2,1)
end
@test t == 71

rng = MersenneTwister(0)
@test gen(mdp, s2, [:up, :up, :up], rng) == (sp=[GWPos(-1,-1), GWPos(-1,-1), GWPos(-1,-1)], r=0.)

# render(mdp, s3)

@test length(states(mdp)) == 100^3

@test decompose_indices(3) == [[1,2], [1,3], [2,3]]
@test length(decompose(mdp)) == 3

