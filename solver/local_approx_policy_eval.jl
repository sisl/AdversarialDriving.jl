using Random
using Printf
using POMDPs
using POMDPModelTools
using GridInterpolations
using LocalFunctionApproximation
using Distributions

import POMDPs: Solver, solve, Policy, action, value

mutable struct LocalPolicyEvalSolver{I<:LocalFunctionApproximator, RNG<:AbstractRNG} <: Solver
    interp::I # Will be copied over by value to each policy
    max_iterations::Int64 # max number of iterations
    belres::Float64 # the Bellman Residual
    verbose::Bool # Whether to print while solving or not
    rng::RNG # Seed if req'd
    is_mdp_generative::Bool # Whether to treat underlying MDP model as generative
    n_generative_samples::Int64 # If underlying model generative, how many samples to use
end

# Default constructor
function LocalPolicyEvalSolver(interp::I;
                                                max_iterations::Int64=100, belres::Float64=1e-3,
                                                verbose::Bool=false, rng::RNG=Random.GLOBAL_RNG,
                                                is_mdp_generative::Bool=false, n_generative_samples::Int64=0) where {I<:LocalFunctionApproximator, RNG<:AbstractRNG}
    return LocalPolicyEvalSolver(interp,max_iterations, belres, verbose, rng, is_mdp_generative, n_generative_samples)
end

# Unparameterized constructor just for getting requirements
function LocalPolicyEvalSolver()
    throw(ArgumentError("LocalPolicyEvalSolver needs a LocalFunctionApproximator object for construction!"))
end


# NOTE : We work directly with the value function
# And extract actions at the end by using the interpolation object
mutable struct LocalPolicyEvalPolicy{I<:LocalFunctionApproximator, RNG<:AbstractRNG} <: Policy
    interp::I # General approximator to be used in VI
    action_map::Vector # Maps the action index to the concrete action type
    mdp::Union{MDP,POMDP} # Uses the model for indexing in the action function
    is_mdp_generative::Bool # (Copied from solver.is_mdp_generative)
    n_generative_samples::Int64 # (Copied from solver.n_generative_samples)
    rng::RNG # (Copied from solver.rng)
end

# The policy can be created using the MDP and solver information
# The policy's function approximation object (interp) is obtained by deep-copying over the
# solver's interp object. The other policy parameters are also obtained from the solver
function LocalPolicyEvalPolicy(mdp::Union{MDP,POMDP},
                                                solver::LocalPolicyEvalSolver)
    return LocalPolicyEvalPolicy(deepcopy(solver.interp),ordered_actions(mdp),mdp,
                                                  solver.is_mdp_generative,solver.n_generative_samples,solver.rng)
end


@POMDP_require solve(solver::LocalPolicyEvalSolver, mdp::Union{MDP,POMDP}) begin

    P = typeof(mdp)
    S = statetype(P)
    A = actiontype(P)
    @req discount(::P)
    @subreq ordered_actions(mdp)

    @req actionindex(::P, ::A)
    @req actions(::P, ::S)
    as = actions(mdp)
    a = first(as)

    @req convert_s(::Type{S}, ::AbstractVector{Float64}, ::P)
    @req convert_s(::Type{Vector{Float64}}, ::S, ::P)

    # Have different requirements depending on whether solver MDP is generative or explicit
    if solver.is_mdp_generative
        @req gen(::DDNOut{(:sp, :r)}, ::P, ::S, ::A, ::typeof(solver.rng))
    else
        @req transition(::P, ::S, ::A)
        pts = get_all_interpolating_points(solver.interp)
        pt = first(pts)
        ss = POMDPs.convert_s(S, pt, mdp)
        dist = transition(mdp, ss, a)
        D = typeof(dist)
        @req support(::D)
    end

end


function POMDPs.solve(solver::LocalPolicyEvalSolver, mdp::Union{MDP,POMDP})

    @warn_requirements solve(solver, mdp)

    # Ensure that generative model has a non-zero number of samples
    if solver.is_mdp_generative
        @assert solver.n_generative_samples > 0
    end

    # Solver parameters
    max_iterations = solver.max_iterations
    belres = solver.belres
    discount_factor = discount(mdp)

    # Initialize the policy
    policy = LocalPolicyEvalPolicy(mdp,solver)

    total_time = 0.0
    iter_time = 0.0

    # Get attributes of interpolator
    # Since the policy object is created by the solver, it directly
    # modifies the value of the interpolator of the created policy
    num_interps = n_interpolating_points(policy.interp)
    interp_points = get_all_interpolating_points(policy.interp)
    interp_values = get_all_interpolating_values(policy.interp)

    # Obtain the vector of states by converting the corresponding
    # vector of interpolation points/samples to the state type
    # using the user-provided convert_s function
    S = statetype(typeof(mdp))
    interp_states = Vector{S}(undef, num_interps)
    for (i, pt) in enumerate(interp_points)
        interp_states[i] = POMDPs.convert_s(S, pt, mdp)
    end

    # State transition dictionary
    next_state_dict = Dict()

    # Outer loop for Value Iteration
    for i = 1 : max_iterations
        residual::Float64 = 0.0
        iter_time = @elapsed begin

        for (istate,s) in enumerate(interp_states)
            sub_aspace = actions(mdp, s)

            if isterminal(mdp, s)
                interp_values[istate] = 0.0
            else
                old_util = interp_values[istate]
                total_util = 0

                for a in sub_aspace
                    u::Float64 = 0.0

                    # Do bellman backup based on generative / explicit model
                    if solver.is_mdp_generative
                        # Generative Model
                        for j in 1:solver.n_generative_samples
                            sp_point, r, isTerm = 0, 0, false
                            if haskey(next_state_dict, (s,a))
                                sp_point, r, isTerm = next_state_dict[(s,a)]
                            else
                                sp, r = gen(DDNOut(:sp,:r), mdp, s, a, solver.rng)
                                isTerm = isterminal(mdp, sp)
                                sp_point = POMDPs.convert_s(Vector{Float64}, sp, mdp)
                                next_state_dict[(s,a)] = (sp_point, r, isTerm)
                            end

                            u += r

                            # Only interpolate sp if it is non-terminal
                            if !isTerm
                                u += discount_factor*compute_value(policy.interp, sp_point)
                            end
                        end
                        u = u / solver.n_generative_samples
                    else
                        # Explicit Model
                        dist = transition(mdp, s, a)
                        for (sp, p) in weighted_iterator(dist)
                            p == 0.0 ? continue : nothing
                            r = reward(mdp, s, a, sp)
                            u += p*r

                        # Only interpolate sp if it is non-terminal
                            if !isterminal(mdp, sp)
                                sp_point = POMDPs.convert_s(Vector{Float64}, sp, mdp)
                                u += p * (discount_factor*LocalFunctionApproximation.compute_value(policy.interp, sp_point))
                            end
                        end # next-states
                    end
                    total_util += action_probability(mdp, s, a)*u
                end #action

                # Update this interpolant value
                interp_values[istate] = total_util
                util_diff = abs(total_util - old_util)
                util_diff > residual ? (residual = util_diff) : nothing
            end
        end #state

        end #time
        total_time += iter_time
        solver.verbose ? @printf("[Iteration %-4d] residual: %10.3G | iteration runtime: %10.3f ms, (%10.3G s total)\n", i, residual, iter_time*1000.0, total_time) : nothing
        residual < belres ? break : nothing

    end #main
    return policy
end


function POMDPs.value(policy::LocalPolicyEvalPolicy, s::S) where S
    # Call the conversion function on the state to get the corresponding vector
    # That represents the point at which to interpolate the function
    s_point = POMDPs.convert_s(Vector{Float64}, s, policy.mdp)
    val = LocalFunctionApproximation.compute_value(policy.interp, s_point)
    return val
end

# Not explicitly stored in policy - extract from value function interpolation
POMDPs.action(policy::LocalPolicyEvalPolicy, s, rng= Random.GLOBAL_RNG) = action_and_prob(policy, s, rng)[1]

function action_and_prob(policy::LocalPolicyEvalPolicy, s, rng= Random.GLOBAL_RNG)
    us = [value(policy, s, a) for a in actions(policy.mdp, s)]
    if sum(us) == 0
        us = ones(length(us)) / length(us)
    end
    us /= sum(us)
    ai = rand(rng, Categorical(us))
    policy.action_map[ai], us[ai]
end

# GlobalApproximationFailureProbe the action-value for some state-action pair
# This is also used in the above function
function POMDPs.value(policy::LocalPolicyEvalPolicy, s::S, a::A) where {S,A}

    mdp = policy.mdp
    discount_factor = discount(mdp)

    u::Float64 = 0.0

    # As in solve(), do different things based on whether
    # mdp is generative or explicit
    if policy.is_mdp_generative
        for j in 1:policy.n_generative_samples
            sp, r = gen(DDNOut(:sp,:r), mdp, s, a, policy.rng)
            sp_point = POMDPs.convert_s(Vector{Float64}, sp, mdp)
            u += r + discount_factor*LocalFunctionApproximation.compute_value(policy.interp, sp_point)
        end
        u = u / policy.n_generative_samples
    else
        dist = transition(mdp, s, a)
        for (sp, p) in weighted_iterator(dist)
            p == 0.0 ? continue : nothing
            r = reward(mdp, s, a, sp)
            u += p*r

            # Only interpolate sp if it is non-terminal
            if !isterminal(mdp, sp)
                sp_point = POMDPs.convert_s(Vector{Float64}, sp, mdp)
                u += p*(discount_factor*LocalFunctionApproximation.compute_value(policy.interp, sp_point))
            end
        end
    end

    return u*action_probability(mdp, s, a)
end

