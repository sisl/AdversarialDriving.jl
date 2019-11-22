include("TIDM.jl")
include("T_intersection_setup.jl")
include("adversarial_pomdp.jl")

function generate_ADM_POMDP()
    roadway, intersection_enter_loc, intersection_exit_loc, goals, should_blink = generate_T_intersection()
    scene = BlinkerScene()

    # Construct the vehicles for the scene
    push!(scene, BV(VecSE2(polar(20.0,-π) - dy, 0), 15., 2, false, 1, roadway))
    push!(scene, BV(VecSE2(polar(35.0,-π) - dy, 0), 15., 2, false, 2, roadway))
    push!(scene, BV(VecSE2(polar(50.0,-π) - dy, 0), 15., 2, false, 3, roadway))
    push!(scene, BV(VecSE2(polar(20.0, 0) + dy, -π), 15., 4, true, 4, roadway))
    push!(scene, BV(VecSE2(polar(15.0,-π/2) + dx, π/2), 10., 5, true,5, roadway))

    # Construct Models for non-ego actors (controlled by the Learner)
    models = Dict{Int, DriverModel}()
    template = generate_TIDM_AST(intersection_enter_loc, intersection_exit_loc, goals, should_blink)
    models[1] = gen_TIDM_AST(template, 0.5, 10)
    models[2] = gen_TIDM_AST(template, 0.5, 10)
    models[3] = gen_TIDM_AST(template, 0.5, 10)
    models[4] = gen_TIDM_AST(template, 0.5, 10)

    # Construct the parameters of the ego vehicle policy
    egoid = 5
    models[egoid] = TIDM(template)
    models[egoid].foce_action = false
    models[egoid].idm = IntelligentDriverModel() # TODO: Add stochacity to the model

    # Simulation timestepping
    timestep = 0.1
    t_end = 10

    AdversarialADM(length(scene), models, roadway, egoid, timestep, t_end, scene)
end

# Sample a fixed number of ADM tasks (in the form of POMDPs)
sample_ADM_POMDPs(n_tasks) = [generate_ADM_POMDP() for i in 1:n_tasks]
