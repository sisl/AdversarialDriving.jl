include("../simulator/adm_task_generator.jl")
using Test

# generate a pomdp
pomdp = generate_ADM_POMDP(dt=0.15, T=15)

# Try the sampling functionality
@test size(sample_ADM_POMDPs(10, dt=1)) == (10,)

# Confirm some things about it
@test o_dim(pomdp) == length(pomdp.initial_scene)*OBS_PER_VEH
@test a_dim(pomdp) == length(pomdp.initial_scene)*ACT_PER_VEH

@test pomdp.num_vehicles == length(pomdp.initial_scene)
@test pomdp.dt == 0.15
@test pomdp.T == 15
@test max_steps(pomdp) == 101

# Confirm initial state
s0 = POMDPs.initialstate(pomdp)
@test s0 == (pomdp.initial_scene, 0.)

# Get the scene and time from the POMDP state
@test get_scene(s0) == pomdp.initial_scene
@test get_t(s0) == 0.

# Discount is set to 1
@test POMDPs.discount(pomdp) == 1

# Check conversion of vehicle state to vector
veh = pomdp.initial_scene[1]
veh1vec = [posg(veh.state).x, posg(veh.state).y, posg(veh.state).θ, vel(veh.state), laneid(veh), veh.state.blinker]
@test to_vec(veh) == veh1vec

# Check conversion of vector to ADM actions
as = fill(LaneFollowingAccelBlinker(0,0,0,false), pomdp.num_vehicles)
@test to_actions(pomdp, zeros(a_dim(pomdp))) == as

# Check the conversion backwards
@test to_vec(as) == vcat([[0., -1., -1.] for i=1:pomdp.num_vehicles]...)

# Observe the entire state
svec = observe_state(pomdp, s0)
@test size(svec) == (OBS_PER_VEH*pomdp.num_vehicles, )
@test svec[1:OBS_PER_VEH] == veh1vec

# The initial scene should have not collisions and is not terminal
@test !iscollision(pomdp, s0)
@test !isterminal(pomdp, s0)

# Generate a set of non-actions for the pomdp
na = nominal_action(pomdp, s0)
avec = to_actions(pomdp, na)
for (i,veh) in enumerate(pomdp.initial_scene)
    @test avec[veh.id].toggle_blinker == false
    @test avec[veh.id].da == 0.
    @test avec[veh.id].toggle_blinker == false
end

# Generate a new state, observation and reward from that action
sp, o, r = gen(pomdp, s0, na)

random_action(pomdp, s0)
mcts_rollout(pomdp, s0)

pomdp = generate_ADM_POMDP(dt=0.3, T=13)
_,_,_,scenes = policy_rollout(pomdp, (o) -> random_action(pomdp, s0), s0, save_scenes = true)

# Uncomment below to get interactive window of previous rollout
# include("../experiments/plot_utils.jl")
# c = plot_scene(get_scene(s0), pomdp.models, pomdp.roadway; egoid = pomdp.egoid)
# write_to_svg(c, "starting_still.svg")

# make_video(scenes, pomdp.models, pomdp.roadway, "scenario.gif"; egoid = pomdp.egoid)

