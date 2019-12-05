include("TIDM.jl")
include("generate_roadway.jl")
include("adm_pomdp.jl")
using Random

function generate_ADM_POMDP(; dt = 0.1, T=10, rng = Random.GLOBAL_RNG)
    roadway, yields_way, intersection_enter_loc, intersection_exit_loc, goals, should_blink, dx, dy = generate_T_intersection()
    scene = BlinkerScene()

    # Construct the vehicles for the scene
    push!(scene, BV(VecSE2(polar(20.0,-π) - dy, 0), 15., goals[2], 2, false, 1, roadway))
    push!(scene, BV(VecSE2(polar(35.0,-π) - dy, 0), 15., goals[2], 2, false, 2, roadway))
    push!(scene, BV(VecSE2(polar(50.0,-π) - dy, 0), 15., goals[2], 2, false, 3, roadway))
    push!(scene, BV(VecSE2(polar(20.0, 0) + dy, -π), 15., goals[4], 4, true, 4, roadway))
    push!(scene, BV(VecSE2(polar(15.0,-π/2) + dx, π/2), 10., goals[5], 5, true,5, roadway))

    # Construct Models for non-ego actors (controlled by the Learner)
    models = Dict{Int, DriverModel}()
    template = generate_TIDM_AST(yields_way, intersection_enter_loc, intersection_exit_loc, goals, should_blink)
    models[1] = generate_TIDM_AST(template, p_toggle_blinker = 1e-4, p_toggle_goal = 1e-4, σ2a = 1)
    models[2] = generate_TIDM_AST(template, p_toggle_blinker = 1e-4, p_toggle_goal = 1e-4, σ2a = 1)
    models[3] = generate_TIDM_AST(template, p_toggle_blinker = 1e-4, p_toggle_goal = 1e-4, σ2a = 1)
    models[4] = generate_TIDM_AST(template, p_toggle_blinker = 1e-4, p_toggle_goal = 1e-4, σ2a = 1)

    # Construct the parameters of the ego vehicle policy
    egoid = 5
    models[egoid] = generate_TIDM_AST(template, p_toggle_blinker = 0., p_toggle_goal = 0., σ2a = 0.)
    models[egoid].force_action = false

    headway_t = max(0.5, rand(rng, Normal(1.5, 0.5))) # desired time headway [s]
    v_des = max(15.0, rand(rng, Normal(29.0, 5.0))) # desired speed [m/s]
    s_min = max(1.0, rand(rng, Normal(5.0, 1.0))) # minimum acceptable gap [m]
    a_max = max(2.0, rand(rng, Normal(3.0, 1.0))) # maximum acceleration ability [m/s²]

    models[egoid].idm = IntelligentDriverModel(T = headway_t, v_des = v_des, s_min = s_min, a_max = a_max)

    # Simulation timestepping
    timestep = dt
    t_end = T

    AdversarialADM(length(scene), models, roadway, egoid, timestep, t_end, scene)
end

# Sample a fixed number of ADM tasks (in the form of POMDPs)
sample_ADM_POMDPs(n_tasks; dt = 0.1, T=10) = [generate_ADM_POMDP(dt=dt, T=T) for i in 1:n_tasks]

