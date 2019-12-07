include("../solver/MAML.jl")
include("../simulator/adm_task_generator.jl")
using Flux: ADAM
using Flux.Tracker: gradient

t = generate_ADM_POMDP()
input_size, output_size = o_dim(t), a_dim(t)
policy = init_policy([input_size, 100, 50, 25, output_size])
params = to_params(policy)
opt = ADAM(0.001, (0.9, 0.999))
N_tasks = 5
N_eps_train = 10
N_eps_test = 1
inner_lr = 0.01

loss() = maml_task_batch_loss(policy, gen_ADM_sampler(dt = 0.3, T=13), N_tasks, N_eps_train, N_eps_test, inner_lr, 0.95, 1., first_order = true)

N = 1
@time for i=1:N
    grads = gradient(() -> loss(), params)
    update_with_clip!(opt, grads, params, max_norm)

    println("Finished epoch, ", i, " return: ", episode_returns(task, policy, 10), " grad norms: ", clipped_grad_norms(grads, params, max_norm))
end

