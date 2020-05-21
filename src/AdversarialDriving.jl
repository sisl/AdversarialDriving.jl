module AdversarialDriving
    using POMDPs
    using AutomotiveSimulator
    using AutomotiveVisualization
    using Distributions
    using Parameters
    using Random

    # TIDM model exports
    export TIDM, BlinkerState, BlinkerVehicle, LaneFollowingAccelBlinker,
           lane_belief, laneid, can_have_goal
    # T-Intersection roadway exports
    export T_intersection, T_int_goal_label, T_int_signal_right
    # MDP
    export AdversarialDrivingMDP, ACTIONS, ACTION_PROB, construct_actions, action_probability
    #helpers
    export decompose_scene, ego, left_straight, left_turnright, right_straight,
           right_turnleft, random_IDM, TIDM_template, Tint_roadway

    include("TIDM.jl")
    include("T_intersection.jl")
    include("mdp.jl")
    include("helpers.jl")
end

