using Flux
using Flux.Tracker
using LinearAlgebra
using Serialization

# Generic training loop that applies clipping and stores training info
function train!(policy, lossfn, N, lr, max_norm, training_log, policy_filename = nothing)
    # Store the return of the policy for later comparison
    lossfn(policy)
    best_return = training_log["return"][end]

    # Define the optimizer with the desired learning rate
    opt = ADAM(lr, (0.9, 0.999))

    # Pull out the vector of parameters for Flux
    params = to_params(policy)
    for i=1:N
        # Store the previous policy so that the KL-Divergence can be computed between steps
        prev_policy = deepcopy(policy)

        # Compute the gradients and update the parameters
        grads = gradient(() -> lossfn(policy), params)
        update_with_clip!(opt, grads, params, max_norm)

        # Store relevant metrics for training (kl divergence and magnitude of clipped gradients)
        add_entry(training_log, "kl", kl_divergence(policy, prev_policy, training_log["last_obs"]))
        add_entry(training_log, "grad_norm", clipped_grad_norms(grads, params, max_norm))
        add_entry(training_log, "lr", lr)

        # Write the best policy to disk
        if !isnothing(policy_filename) && training_log["return"][end] > best_return
            println("Saving new best policy policy!")
            best_return = training_log["return"][end]
            save_policy(policy_filename, policy)
        end
        if haskey(training_log, "return_before")
            println("e=", i, " return_before: ", training_log["return_before"][end], " max_ret_before: ", training_log["max_return_before"][end], " return_after: ", training_log["return_after"][end], " max_ret_after: ", training_log["max_return_after"][end], " gnorm: ", training_log["grad_norm"][end], " kl: ", training_log["kl"][end], " stdev: ", mean(policy.weights["σ2"]) )
        else
            println("e=", i, " avg_ret: ", training_log["return"][end], " max_ret: ", training_log["max_return"][end], " gnorm: ", training_log["grad_norm"][end], " kl: ", training_log["kl"][end], " stdev: ", mean(policy.weights["σ2"]) )
        end
        serialize(string(policy_filename, ".training_log"), training_log)
    end
end

function train_while_dropping_lr(policy, lossfn, N, lr, max_norm, training_log, policy_filename; scale = 0.2)
    save_policy(policy_filename, policy)
    lr = lr / scale
    while lr > 1e-7
        lr = lr*scale
        println("Setting the learning rate to lr=", lr)
        policy = load_policy(policy_filename)
        train!(policy, lossfn, N, lr, max_norm, training_log, policy_filename)
    end
end

# Training loop
function train_with_restarts(policy, lossfn, N, lr, max_norm, training_log, policy_filename; scale = 0.2)
    # Save the initial policy
    save_policy(policy_filename, policy)
    lr = lr / scale
    while true
        # Keep a reference policy
        prev_policy_outer = load_policy(policy_filename)
        # Drop the learning rate
        lr = lr*scale
        println("Setting the learning rate to lr=", lr)

        # Train until there is no more improvement
        # while true
            println("Training for N=", N, " steps")
            # prev_policy_inner = load_policy(policy_filename)
            policy_to_update = load_policy(policy_filename)
            train!(policy_to_update, lossfn, N, lr, max_norm, training_log, policy_filename)
            # (prev_policy_inner == load_policy(policy_filename)) && break
        # end

        println("Found no further improvement at this learning rate")

        # If the best policy after training was just the old policy then we don't have any improvement so we should stop
        prev_policy_outer == load_policy(policy_filename) && break
    end
    println("Done training, no further improvement found")
end

# Add entry to a log dictionary. If no key exists make new array, otherwise push to array
function add_entry(d::Dict, key, val)
    !haskey(d, key) ? (d[key] = [val]) : push!(d[key], val)
end

# Computes the norm of the gradients
function grad_norms(grads, params, p=2)
    gnorm = 0
    for param in params
        gnorm += sum(grads[param].data .^ p)
    end
    gnorm ^ (1. / p)
end

# Clips the norm of the gradient to max norm
# Returns the scaling factor to multiply the gradient to
function clip_norm_scale(grads, params, max_norm, p=2)
    gnorm = grad_norms(grads, params, p)
    gnorm > max_norm ? max_norm/gnorm : 1
end

# Returns the clipped value of the gradient norms
# used for diagnosing
clipped_grad_norms(grads, params, max_norm, p=2) = min(max_norm, grad_norms(grads, params, p))

# update the parameters with gradient clipping
function update_with_clip!(opt, grads, params, max_norm, p=2)
    scale = clip_norm_scale(grads, params, max_norm, p)
    for p in params
        Flux.Tracker.update!(opt, p, scale*grads[p])
    end
end

