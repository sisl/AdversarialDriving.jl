include("TIDM.jl")
include("generate_roadway.jl")
include("adm_pomdp.jl")
using Random

function generate_2car_scene(;dt = 0.1, rng = Random.GLOBAL_RNG)
    # Create the roadway
    roadway, yields_way, intersection_enter_loc, intersection_exit_loc, goals, should_blink, dx, dy = generate_T_intersection()
    template = generate_TIDM_AST(yields_way, intersection_enter_loc, intersection_exit_loc, goals, should_blink)
    support = [0, -3, -1.5, 1.5, 3]
    p_big = 1e-2
    p_small = 1e-1

    stochastic_probs = [1 - (2*p_big + 2*p_small), p_big, p_small, p_small, p_big]
    deterministic_probs = [1.0, 0., 0., 0., 0.]


    # Create the ego vehicle
    egoid = 2
    egomodel = generate_TIDM_AST(template, p_toggle_blinker = 0., p_toggle_goal = 0., da_dist = DiscreteNonParametric(support, deterministic_probs))
    egomodel.force_action = false
    egovehicle = BV(VecSE2(polar(15.0,-π/2) + dx, π/2), 9., goals[5], 5, true, egoid, roadway)

    # create other vehicle and model
    vehicle = BV(VecSE2(polar(29.0,-π) - dy, 0), 10., goals[2], 2, false, 1, roadway)
    p_blink = 1e-5
    p_goal= 1e-5
    model = generate_TIDM_AST(template, p_toggle_blinker = p_blink, p_toggle_goal = p_goal, da_dist = DiscreteNonParametric(support, stochastic_probs))

    scene = BlinkerScene()
    push!(scene, vehicle)
    push!(scene, egovehicle)
    models = Dict{Int, DriverModel}(1 => model, 2=>egomodel)

    AdversarialADM(models, roadway, egoid, scene, dt)
end



function generate_decomposed_scene(;dt = 0.1, rng = Random.GLOBAL_RNG)
    # Create the roadway
    roadway, yields_way, intersection_enter_loc, intersection_exit_loc, goals, should_blink, dx, dy = generate_T_intersection()
    template = generate_TIDM_AST(yields_way, intersection_enter_loc, intersection_exit_loc, goals, should_blink)
    support = [0, -3, -1.5, 1.5, 3]
    p_big = 1e-2
    p_small = 1e-1

    stochastic_probs = [1 - (2*p_big + 2*p_small), p_big, p_small, p_small, p_big]
    deterministic_probs = [1.0, 0., 0., 0., 0., 0.]


    # Create the ego vehicle
    egoid = 2
    egomodel = generate_TIDM_AST(template, p_toggle_blinker = 0., p_toggle_goal = 0., da_dist = DiscreteNonParametric(support, deterministic_probs))
    egomodel.force_action = false
    # headway_t = max(0.5, rand(rng, Normal(1.5, 0.5))) # desired time headway [s]
    # v_des = max(15.0, rand(rng, Normal(20.0, 5.0))) # desired speed [m/s]
    # s_min = max(1.0, rand(rng, Normal(5.0, 1.0))) # minimum acceptable gap [m]
    # a_max = max(2.0, rand(rng, Normal(3.0, 1.0))) # maximum acceleration ability [m/s²]
    # egomodel.idm = IntelligentDriverModel(T = headway_t, v_des = v_des, s_min = s_min, a_max = a_max)
    egovehicle = BV(VecSE2(polar(15.0,-π/2) + dx, π/2), 9., goals[5], 5, true, egoid, roadway)

    # create list of other vehicles and models
    vehicles = [BV(VecSE2(polar(20.0, 0) + dy, -π), 20., goals[3], 3, false, 1, roadway),
                BV(VecSE2(polar(35.0,-π) - dy, 0), 15., goals[2], 2, false, 1, roadway),
                BV(VecSE2(polar(50.0,-π) - dy, 0), 20., goals[2], 2, false, 1, roadway),
                BV(VecSE2(polar(10.0, 0) + dy, -π), 15., goals[4], 4, true, 1, roadway),
                ]
    p_blink = 1e-5
    p_goal= 1e-5
    models = [generate_TIDM_AST(template, p_toggle_blinker = p_blink, p_toggle_goal = p_goal, da_dist = DiscreteNonParametric(support, stochastic_probs)),
              generate_TIDM_AST(template, p_toggle_blinker = p_blink, p_toggle_goal = p_goal, da_dist = DiscreteNonParametric(support, stochastic_probs)),
              generate_TIDM_AST(template, p_toggle_blinker = p_blink, p_toggle_goal = p_goal, da_dist = DiscreteNonParametric(support, stochastic_probs)),
              generate_TIDM_AST(template, p_toggle_blinker = p_blink, p_toggle_goal = p_goal, da_dist = DiscreteNonParametric(support, stochastic_probs))]

    combined_scene = BlinkerScene()
    combined_models = Dict{Int, DriverModel}()

    decomposed_pomdps = []
    # Create the mdps
    for i = 1:length(vehicles)
        # Create the local mdp
        iscene = BlinkerScene()
        push!(iscene, vehicles[i])
        push!(iscene, egovehicle)
        imodels = Dict{Int, DriverModel}()
        imodels[1] = deepcopy(models[i])
        imodels[2] = deepcopy(egomodel)

        push!(decomposed_pomdps, AdversarialADM(imodels, roadway, egoid, iscene, dt))

        # Store vehicle and models into combined pomdp
        push!(combined_scene, set_veh_id(vehicles[i], i))
        combined_models[i] = deepcopy(models[i])
    end

    # add the ego vehicle to the combined simulator
    egoid = length(vehicles) + 1
    push!(combined_scene, set_veh_id(egovehicle, egoid))
    combined_models[egoid] = deepcopy(egomodel)

    return decomposed_pomdps, AdversarialADM(combined_models, roadway, egoid, combined_scene, dt)
end

