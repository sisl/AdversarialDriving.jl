include("../simulator/task_generator.jl")
using MCTS

pomdp = generate_POMDP()

s0 = (scene, 0.)
# s = deepcopy(s0)
# scenes = [scene]
#
# while !isterminal(pomdp, s)
#     actions = nominal_actions(get_scene(s), pomdp.num_vehicles)
#     global s, o, r = gen(pomdp, s, actions)
#     println(r)
#     push!(scenes, get_scene(s))
# end

solver = DPWSolver(n_iterations=1000, depth=1000, next_action = random_action, k_state = .9, alpha_state = 0., check_repeat_state = false, estimate_value = perform_rollout)

policy = solve(solver, pomdp)
action(policy, s0)
policy.tree




