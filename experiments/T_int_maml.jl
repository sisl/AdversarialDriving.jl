include("../solver/MAML.jl")
include("../simulator/adm_task_generator.jl")
using Flux.Tracker: update!
using Flux: ADAM, params

t = generate_ADM_POMDP()
input_size, output_size = o_dim(t), a_dim(t)
policy = init_policy([input_size, 25, output_size])
opt = ADAM(0.001, (0.9, 0.999))
loss() = maml_task_batch_loss(policy = policy, task_sampler = sample_ADM_POMDPs, N_tasks = 5, N_eps_train = 10, N_eps_test = 1, inner_lr = 1e-6, λ = 0.5, baseline_reg_coeff = 1., first_order = false)

N = 1
@time for i=1:N
    # println("here")
    ps = params(policy)
    grads = gradient(() -> loss(), ps)
    # println("here!")
    for p in ps
        # prinln("a")
        update!(opt, p, grads[p])
    end
    # println("loss: ", loss())
end

